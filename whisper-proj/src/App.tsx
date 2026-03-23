import { useCallback, useEffect, useRef, useState } from 'react'
import { ConsoleWindow } from './components/ConsoleWindow'
import { VoiceControls } from './components/VoiceControls'
import { useAudioRecorder } from './hooks/useAudioRecorder'
import { useWakeWord } from './hooks/useWakeWord'
import { streamChatReply } from './services/chatService'
import { transcribeAudio } from './services/whisperService'
import type { AppStatus, ChatMessage, ChatRole } from './types'
import './App.css'

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

  const wakeWordSupportRef = useRef(true)
  const bootMessageAddedRef = useRef(false)
  const unsupportedMessageAddedRef = useRef(false)

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

  const replaceMessageContent = useCallback((messageId: string, nextContent: string) => {
    setMessages((previousMessages) =>
      previousMessages.map((message) =>
        message.id === messageId
          ? {
              ...message,
              content: nextContent,
            }
          : message,
      ),
    )
  }, [])

  const handleRecorderError = useCallback(
    (message: string) => {
      addMessage('wilga', `[BŁĄD] ${message}`)
      setStatus(wakeWordSupportRef.current ? 'LISTENING_WAKE_WORD' : 'IDLE')
    },
    [addMessage],
  )

  const handleRecordingComplete = useCallback(
    async (audioBlob: Blob) => {
      const nextIdleStatus: AppStatus = wakeWordSupportRef.current ? 'LISTENING_WAKE_WORD' : 'IDLE'

      if (audioBlob.size === 0) {
        addMessage('wilga', '[BŁĄD] Nagranie jest puste. Spróbuj ponownie.')
        setStatus(nextIdleStatus)
        return
      }

      setStatus('PROCESSING')

      try {
        const transcript = await transcribeAudio(audioBlob)
        addMessage('user', transcript)

        const assistantMessage = createMessage('wilga', '')
        setMessages((previousMessages) => [...previousMessages, assistantMessage])

        let chunkReceived = false
        await streamChatReply(transcript, {
          onChunk: (chunk) => {
            chunkReceived = true
            appendToMessage(assistantMessage.id, chunk)
          },
        })

        if (!chunkReceived) {
          replaceMessageContent(assistantMessage.id, '[BŁĄD] Model nie zwrócił treści odpowiedzi.')
        }
      } catch (error) {
        addMessage('wilga', `[BŁĄD] ${toMessage(error)}`)
      } finally {
        setStatus(nextIdleStatus)
      }
    },
    [addMessage, appendToMessage, replaceMessageContent],
  )

  const { isRecording, startRecording, stopRecording } = useAudioRecorder({
    onRecordingComplete: handleRecordingComplete,
    onError: handleRecorderError,
  })

  const startRecordingFlow = useCallback(
    async () => {
      if (status === 'RECORDING' || status === 'PROCESSING') {
        return
      }

      setStatus('RECORDING')

      try {
        await startRecording()
      } catch (error) {
        addMessage('wilga', `[BŁĄD] ${toMessage(error)}`)
        setStatus(wakeWordSupportRef.current ? 'LISTENING_WAKE_WORD' : 'IDLE')
      }
    },
    [addMessage, startRecording, status],
  )

  const handleWakeWordDetected = useCallback(() => {
    void startRecordingFlow()
  }, [startRecordingFlow])

  const {
    isSupported: isWakeWordSupported,
    isListening: isWakeWordListening,
  } = useWakeWord({
    enabled: status === 'LISTENING_WAKE_WORD',
    onWakeWordDetected: handleWakeWordDetected,
    onError: handleRecorderError,
  })

  useEffect(() => {
    if (bootMessageAddedRef.current) {
      return
    }

    bootMessageAddedRef.current = true
    addMessage('wilga', 'Gotowa. Powiedz "hej wilga" albo użyj przycisku, aby rozpocząć.')
  }, [addMessage])

  useEffect(() => {
    wakeWordSupportRef.current = isWakeWordSupported
    if (isWakeWordSupported) {
      return
    }

    if (!unsupportedMessageAddedRef.current) {
      unsupportedMessageAddedRef.current = true
      addMessage(
        'wilga',
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
    if (status === 'PROCESSING') {
      return
    }

    if (isRecording || status === 'RECORDING') {
      stopRecording()
      return
    }

    void startRecordingFlow()
  }, [isRecording, startRecordingFlow, status, stopRecording])

  return (
    <main className="app-shell">
      <header className="app-header">
        <p className="app-kicker">Whisper + Ollama Streaming</p>
        <h1 className="app-title">Voice-to-Text Console</h1>
      </header>

      <VoiceControls
        status={status}
        isRecording={isRecording}
        isWakeWordSupported={isWakeWordSupported}
        isWakeWordListening={isWakeWordListening}
        onToggleRecording={handleToggleRecording}
      />

      <ConsoleWindow messages={messages} />
    </main>
  )
}

export default App
