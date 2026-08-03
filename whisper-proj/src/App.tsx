import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatTranscript } from './components/ChatTranscript'
import { VoiceControls } from './components/VoiceControls'
import { WilgusAssistant } from './components/WilgusAssistant'
import { useAudioRecorder } from './hooks/useAudioRecorder'
import { useFollowUpListener } from './hooks/useFollowUpListener'
import { useMicDevices } from './hooks/useMicDevices'
import { useStudentReady } from './hooks/useStudentReady'
import { useWakeWord } from './hooks/useWakeWord'
import { playAck, stopAck } from './services/ackService'
import {
  CHAT_HISTORY_MAX_MESSAGES,
  streamChatReply,
  type ChatHistoryItem,
} from './services/chatService'
import { startNewSession } from './services/sessionService'
import { extractSentences, SpeechQueue } from './services/speechQueue'
import { transcribeAudio } from './services/whisperService'
import type { AppStatus, ChatMessage, ChatRole } from './types'
import { stripLeadingWakePhrase } from './utils/wakePhrase'
import './App.css'

const BOOT_MESSAGE =
  'Cześć! Jestem Wilguś. Powiedz „hej wilguś” albo kliknij mnie, żeby zacząć.'

/** Default matches backend SESSION_IDLE_ROTATE_SECONDS. */
const DEFAULT_IDLE_ROTATE_MS = 600_000

const buildChatHistory = (messages: ChatMessage[]): ChatHistoryItem[] =>
  messages
    .filter((message) => message.role === 'user' || message.role === 'wilga')
    .filter((message) => message.content.trim().length > 0)
    .filter((message) => message.content.trim() !== BOOT_MESSAGE)
    .slice(-CHAT_HISTORY_MAX_MESSAGES)
    .map((message) => ({
      role: message.role === 'wilga' ? ('assistant' as const) : ('user' as const),
      content: message.content.trim(),
    }))

const now = () =>
  new Date().toLocaleTimeString('pl-PL', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })

const createMessage = (role: ChatRole, content: string): ChatMessage => {
  const fallbackId = `${Date.now()}-${Math.random().toString(16).slice(2)}`
  const id = globalThis.crypto?.randomUUID?.() ?? fallbackId
  return {
    id,
    timestamp: now(),
    role,
    content,
  }
}

const toMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Wystąpił nieoczekiwany błąd.'

function App() {
  const [status, setStatus] = useState<AppStatus>('LISTENING_WAKE_WORD')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [processingHint, setProcessingHint] = useState<string | null>(null)
  const [followUpSecondsLeft, setFollowUpSecondsLeft] = useState<number | null>(null)
  const {
    status: studentStatus,
    error: studentError,
    backendReachable,
    studentReady,
    markWarming,
    refreshNow,
  } = useStudentReady()

  const wakeWordSupportRef = useRef(true)
  const bootMessageAddedRef = useRef(false)
  const unsupportedMessageAddedRef = useRef(false)
  const speechQueueRef = useRef<SpeechQueue | null>(null)
  const streamFinishedRef = useRef(false)
  const awaitingSpeechIdleRef = useRef(false)
  const enterFollowUpAfterReplyRef = useRef(false)
  const statusRef = useRef<AppStatus>(status)
  const lastActivityAtRef = useRef(Date.now())
  const idleRotateMsRef = useRef(DEFAULT_IDLE_ROTATE_MS)
  const sessionRotateInFlightRef = useRef(false)

  useEffect(() => {
    statusRef.current = status
  }, [status])

  const touchActivity = useCallback(() => {
    lastActivityAtRef.current = Date.now()
  }, [])

  const idleStatus = useCallback(
    (): AppStatus => (wakeWordSupportRef.current ? 'LISTENING_WAKE_WORD' : 'IDLE'),
    [],
  )

  const goIdle = useCallback(() => {
    enterFollowUpAfterReplyRef.current = false
    setProcessingHint(null)
    setFollowUpSecondsLeft(null)
    stopAck()
    setStatus(idleStatus())
  }, [idleStatus])

  const goFollowUp = useCallback(() => {
    enterFollowUpAfterReplyRef.current = false
    setProcessingHint(null)
    setFollowUpSecondsLeft(30)
    setStatus('LISTENING_FOLLOW_UP')
  }, [])

  const addMessage = useCallback((role: ChatRole, content: string) => {
    setMessages((previousMessages) => [...previousMessages, createMessage(role, content)])
  }, [])

  const appendToMessage = useCallback((messageId: string, chunk: string) => {
    setMessages((previousMessages) =>
      previousMessages.map((message) =>
        message.id === messageId
          ? {
              ...message,
              content: `${message.content}${chunk}`,
            }
          : message,
      ),
    )
  }, [])

  useEffect(() => {
    const queue = new SpeechQueue({
      onSpeakingStart: () => {
        stopAck()
        setProcessingHint(null)
        setStatus('SPEAKING')
      },
      onIdle: () => {
        if (streamFinishedRef.current && awaitingSpeechIdleRef.current) {
          awaitingSpeechIdleRef.current = false
          if (enterFollowUpAfterReplyRef.current) {
            goFollowUp()
          } else {
            goIdle()
          }
        }
      },
      onError: (message) => {
        addMessage('system', `[BŁĄD] TTS: ${message}`)
      },
    })
    speechQueueRef.current = queue
    return () => {
      queue.cancel()
      speechQueueRef.current = null
    }
  }, [addMessage, goFollowUp, goIdle])

  const handleRecorderError = useCallback(
    (message: string) => {
      speechQueueRef.current?.cancel()
      awaitingSpeechIdleRef.current = false
      streamFinishedRef.current = true
      addMessage('system', `[BŁĄD] ${message}`)
      goIdle()
    },
    [addMessage, goIdle],
  )

  /** Wake-word soft/hard notices — never cancel TTS or spam restart loops. */
  const handleWakeWordError = useCallback(
    (message: string) => {
      addMessage('system', `[INFO] ${message}`)
    },
    [addMessage],
  )

  const handleRecordingComplete = useCallback(
    async (audioBlob: Blob) => {
      touchActivity()
      if (audioBlob.size === 0) {
        addMessage('system', '[BŁĄD] Nagranie jest puste. Spróbuj ponownie.')
        goIdle()
        return
      }

      speechQueueRef.current?.reset()
      streamFinishedRef.current = false
      awaitingSpeechIdleRef.current = false
      enterFollowUpAfterReplyRef.current = false
      setFollowUpSecondsLeft(null)
      setStatus('PROCESSING')

      let speechBuffer = ''

      const flushSpeech = (final = false) => {
        const queue = speechQueueRef.current
        if (!queue) {
          return
        }
        if (final) {
          const leftover = speechBuffer.trim()
          speechBuffer = ''
          if (leftover) {
            queue.enqueue(leftover)
          }
          return
        }
        const { sentences, rest } = extractSentences(speechBuffer)
        speechBuffer = rest
        for (const sentence of sentences) {
          queue.enqueue(sentence)
        }
      }

      try {
        const rawTranscript = await transcribeAudio(audioBlob)
        const transcript = stripLeadingWakePhrase(rawTranscript)
        if (!transcript) {
          stopAck()
          setProcessingHint(null)
          addMessage(
            'system',
            '[INFO] Usłyszałam tylko komendę aktywacji. Po „hej wilguś” od razu zadaj pytanie.',
          )
          goIdle()
          return
        }

        // Instant filler while STT→chat→TTS runs in parallel.
        setProcessingHint(playAck())

        // History from state before this turn (current question goes as `message`).
        const history = buildChatHistory(messages)
        addMessage('user', transcript)

        let chunkReceived = false
        let streamError: string | null = null
        let assistantMessageId: string | null = null

        await streamChatReply(transcript, {
          history,
          onChunk: (chunk) => {
            if (!assistantMessageId) {
              const assistantMessage = createMessage('wilga', '')
              assistantMessageId = assistantMessage.id
              setMessages((previousMessages) => [...previousMessages, assistantMessage])
            }
            chunkReceived = true
            appendToMessage(assistantMessageId, chunk)
            speechBuffer += chunk
            flushSpeech(false)
          },
          onError: (message) => {
            streamError = message
          },
        })

        flushSpeech(true)
        streamFinishedRef.current = true

        if (streamError) {
          addMessage('system', `[BŁĄD] ${streamError}`)
        }

        if (!chunkReceived && !streamError) {
          stopAck()
          setProcessingHint(null)
          addMessage('system', '[BŁĄD] Model nie zwrócił treści odpowiedzi.')
          goIdle()
          return
        }

        const shouldFollowUp = chunkReceived && !streamError
        enterFollowUpAfterReplyRef.current = shouldFollowUp

        const queue = speechQueueRef.current
        if (queue?.isBusy) {
          awaitingSpeechIdleRef.current = true
        } else if (shouldFollowUp) {
          stopAck()
          goFollowUp()
        } else {
          stopAck()
          goIdle()
        }
      } catch (error) {
        speechQueueRef.current?.cancel()
        awaitingSpeechIdleRef.current = false
        streamFinishedRef.current = true
        enterFollowUpAfterReplyRef.current = false
        stopAck()
        setProcessingHint(null)
        const detail = toMessage(error)
        if (detail.toLowerCase().includes('transcription is empty')) {
          addMessage(
            'system',
            '[INFO] Nagranie było puste lub za krótkie. Po „hej wilguś” mów od razu pytanie.',
          )
        } else {
          addMessage('system', `[BŁĄD] ${detail}`)
        }
        goIdle()
      }
    },
    [addMessage, appendToMessage, goFollowUp, goIdle, messages, touchActivity],
  )

  const beginNewSession = useCallback(
    async (reason: 'manual' | 'idle') => {
      if (sessionRotateInFlightRef.current) {
        return
      }
      if (
        statusRef.current === 'RECORDING' ||
        statusRef.current === 'PROCESSING' ||
        statusRef.current === 'SPEAKING'
      ) {
        return
      }

      sessionRotateInFlightRef.current = true
      speechQueueRef.current?.cancel()
      awaitingSpeechIdleRef.current = false
      streamFinishedRef.current = true
      stopAck()
      setProcessingHint(null)
      setFollowUpSecondsLeft(null)
      enterFollowUpAfterReplyRef.current = false
      markWarming()

      const infoLabel =
        reason === 'idle'
          ? '[INFO] Brak aktywności — nowa rozmowa, rozgrzewam model…'
          : '[INFO] Nowa rozmowa — rozgrzewam model…'

      setMessages([
        createMessage('wilga', BOOT_MESSAGE),
        createMessage('system', infoLabel),
      ])
      setStatus(idleStatus())

      try {
        const result = await startNewSession()
        if (result.idle_rotate_seconds > 0) {
          idleRotateMsRef.current = result.idle_rotate_seconds * 1000
        }
        touchActivity()
        refreshNow()
        addMessage(
          'system',
          result.advanced
            ? '[INFO] Kontekst domu odświeżony — czekam na rozgrzanie modelu.'
            : '[INFO] Sesja zresetowana — czekam na rozgrzanie modelu.',
        )
      } catch (error) {
        addMessage('system', `[BŁĄD] ${toMessage(error)}`)
        refreshNow()
      } finally {
        sessionRotateInFlightRef.current = false
      }
    },
    [addMessage, idleStatus, markWarming, refreshNow, touchActivity],
  )

  const handleNewConversation = useCallback(() => {
    void beginNewSession('manual')
  }, [beginNewSession])

  // Auto new session after idle (same path as „Nowa rozmowa”).
  useEffect(() => {
    const timerId = window.setInterval(() => {
      if (!studentReady || sessionRotateInFlightRef.current) {
        return
      }
      if (
        statusRef.current === 'RECORDING' ||
        statusRef.current === 'PROCESSING' ||
        statusRef.current === 'SPEAKING'
      ) {
        return
      }
      const idleMs = Date.now() - lastActivityAtRef.current
      if (idleMs >= idleRotateMsRef.current) {
        void beginNewSession('idle')
      }
    }, 15_000)
    return () => {
      window.clearInterval(timerId)
    }
  }, [beginNewSession, studentReady])

  const {
    devices,
    deviceId,
    setDeviceId,
    error: micDevicesError,
  } = useMicDevices()

  const {
    isRecording,
    inputLevel,
    startRecording,
    stopRecording,
  } = useAudioRecorder({
    onRecordingComplete: handleRecordingComplete,
    onError: handleRecorderError,
    deviceId,
  })

  useEffect(() => {
    if (micDevicesError) {
      addMessage('system', `[INFO] Mikrofon: ${micDevicesError}`)
    }
  }, [addMessage, micDevicesError])

  const startRecordingFlow = useCallback(
    async () => {
      if (!studentReady) {
        return
      }

      const current = statusRef.current
      if (
        current === 'RECORDING' ||
        current === 'PROCESSING' ||
        current === 'SPEAKING'
      ) {
        return
      }

      touchActivity()
      speechQueueRef.current?.cancel()
      awaitingSpeechIdleRef.current = false
      enterFollowUpAfterReplyRef.current = false
      setFollowUpSecondsLeft(null)
      stopAck()
      setProcessingHint(null)
      setStatus('RECORDING')

      try {
        await startRecording()
      } catch (error) {
        addMessage('system', `[BŁĄD] ${toMessage(error)}`)
        goIdle()
      }
    },
    [addMessage, goIdle, startRecording, studentReady, touchActivity],
  )

  const handleWakeWordDetected = useCallback(() => {
    // Let Chrome release the Web Speech mic before MediaRecorder opens a new stream.
    window.setTimeout(() => {
      void startRecordingFlow()
    }, 350)
  }, [startRecordingFlow])

  const handleFollowUpSpeech = useCallback(() => {
    // Brief gap so follow-up mic tracks can stop before MediaRecorder opens.
    window.setTimeout(() => {
      void startRecordingFlow()
    }, 80)
  }, [startRecordingFlow])

  const handleFollowUpTimeout = useCallback(() => {
    goIdle()
  }, [goIdle])

  const handleFollowUpTick = useCallback((secondsLeft: number) => {
    setFollowUpSecondsLeft(secondsLeft)
  }, [])

  useFollowUpListener({
    enabled: studentReady && status === 'LISTENING_FOLLOW_UP',
    deviceId,
    onSpeechDetected: handleFollowUpSpeech,
    onTimeout: handleFollowUpTimeout,
    onTick: handleFollowUpTick,
  })

  const {
    isSupported: isWakeWordSupported,
    isListening: isWakeWordListening,
  } = useWakeWord({
    enabled: studentReady && status === 'LISTENING_WAKE_WORD',
    onWakeWordDetected: handleWakeWordDetected,
    onError: handleWakeWordError,
  })

  useEffect(() => {
    if (bootMessageAddedRef.current) {
      return
    }

    bootMessageAddedRef.current = true
    addMessage('wilga', BOOT_MESSAGE)
  }, [addMessage])

  useEffect(() => {
    wakeWordSupportRef.current = isWakeWordSupported
    if (isWakeWordSupported) {
      return
    }

    if (!unsupportedMessageAddedRef.current) {
      unsupportedMessageAddedRef.current = true
      addMessage(
        'system',
        '[BŁĄD] Web Speech API jest niedostępne. Wake word wyłączony, działa tylko tryb ręczny.',
      )
    }

    setStatus((currentStatus) => {
      if (currentStatus === 'LISTENING_WAKE_WORD') {
        return 'IDLE'
      }
      return currentStatus
    })
  }, [addMessage, isWakeWordSupported])

  const handleToggleRecording = useCallback(() => {
    if (!studentReady || status === 'PROCESSING' || status === 'SPEAKING') {
      return
    }

    if (isRecording || status === 'RECORDING') {
      stopRecording()
      return
    }

    void startRecordingFlow()
  }, [isRecording, startRecordingFlow, status, stopRecording, studentReady])

  const latestWilgaText =
    [...messages].reverse().find((message) => message.role === 'wilga' && message.content.trim())
      ?.content ?? null

  let bubbleText: string | null = null
  if (!studentReady) {
    bubbleText =
      !backendReachable || studentStatus === 'error'
        ? 'Jeszcze nie mogę rozmawiać — czekam na model…'
        : 'Rozgrzewam model… zaraz będę gotowy.'
  } else if (status === 'PROCESSING') {
    bubbleText = processingHint || 'Hmm, myślę…'
  } else if (status === 'SPEAKING') {
    bubbleText = latestWilgaText
  } else if (
    status === 'LISTENING_WAKE_WORD' ||
    status === 'IDLE' ||
    status === 'LISTENING_FOLLOW_UP'
  ) {
    bubbleText = latestWilgaText
  }

  const wilgusDisabled = !studentReady || status === 'PROCESSING' || status === 'SPEAKING'

  return (
    <main className="app-shell">
      <div className="app-frame">
        <header className="app-header">
          <div>
            <p className="app-kicker">Asystent głosowy domu</p>
            <h1 className="app-title">Wilguś</h1>
          </div>
          <VoiceControls
            status={status}
            isWakeWordSupported={isWakeWordSupported}
            isWakeWordListening={isWakeWordListening}
            devices={devices}
            deviceId={deviceId}
            onDeviceChange={setDeviceId}
            onNewConversation={handleNewConversation}
            followUpSecondsLeft={followUpSecondsLeft}
            studentStatus={studentStatus}
            studentReady={studentReady}
            studentError={studentError}
            backendReachable={backendReachable}
          />
        </header>

        <div className="app-main">
          <ChatTranscript messages={messages} />
          <WilgusAssistant
            status={status}
            bubbleText={bubbleText}
            disabled={wilgusDisabled}
            studentReady={studentReady}
            onClick={handleToggleRecording}
            inputLevel={inputLevel}
          />
        </div>
      </div>
    </main>
  )
}

export default App
