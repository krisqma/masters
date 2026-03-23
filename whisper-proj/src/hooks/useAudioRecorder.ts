import { useCallback, useEffect, useRef, useState } from 'react'

interface UseAudioRecorderOptions {
  onRecordingComplete: (audioBlob: Blob) => void | Promise<void>
  onError?: (message: string) => void
  silenceDurationMs?: number
  silenceThreshold?: number
}

interface UseAudioRecorderResult {
  isRecording: boolean
  startRecording: () => Promise<void>
  stopRecording: () => void
}

const DEFAULT_SILENCE_DURATION_MS = 2_000
const DEFAULT_SILENCE_THRESHOLD = 0.02
const CHUNK_DURATION_MS = 250
const MIME_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']

type WindowWithWebkitAudioContext = Window & {
  webkitAudioContext?: typeof AudioContext
}

const resolveMimeType = () => {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return ''
  }

  for (const mimeType of MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return mimeType
    }
  }

  return ''
}

const toMessage = (error: unknown, fallback: string) =>
  error instanceof Error ? error.message : fallback

export function useAudioRecorder({
  onRecordingComplete,
  onError,
  silenceDurationMs = DEFAULT_SILENCE_DURATION_MS,
  silenceThreshold = DEFAULT_SILENCE_THRESHOLD,
}: UseAudioRecorderOptions): UseAudioRecorderResult {
  const [isRecording, setIsRecording] = useState(false)
  const isRecordingRef = useRef(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const analyserDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null)
  const monitorSilenceRef = useRef<(() => void) | null>(null)
  const silenceStartedAtRef = useRef<number | null>(null)
  const rafIdRef = useRef<number | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const isStoppingRef = useRef(false)
  const onRecordingCompleteRef = useRef(onRecordingComplete)
  const onErrorRef = useRef(onError)

  useEffect(() => {
    onRecordingCompleteRef.current = onRecordingComplete
  }, [onRecordingComplete])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  const cleanupStream = useCallback(() => {
    if (rafIdRef.current !== null) {
      window.cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }

    silenceStartedAtRef.current = null
    analyserDataRef.current = null
    monitorSilenceRef.current = null
    analyserRef.current?.disconnect()
    analyserRef.current = null

    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => undefined)
      audioContextRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop()
      })
      streamRef.current = null
    }
  }, [])

  const stopRecording = useCallback(() => {
    const mediaRecorder = mediaRecorderRef.current
    if (!mediaRecorder || mediaRecorder.state === 'inactive' || isStoppingRef.current) {
      return
    }

    isStoppingRef.current = true
    mediaRecorder.stop()
    isRecordingRef.current = false
    setIsRecording(false)
  }, [])

  const startRecording = useCallback(async () => {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('Przeglądarka nie obsługuje dostępu do mikrofonu.')
    }
    if (typeof MediaRecorder === 'undefined') {
      throw new Error('Przeglądarka nie obsługuje nagrywania audio (MediaRecorder).')
    }
    if (isRecordingRef.current) {
      return
    }

    chunksRef.current = []
    isStoppingRef.current = false

    let stream: MediaStream | null = null
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mimeType = resolveMimeType()
      const mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onerror = () => {
        onErrorRef.current?.('Wystąpił błąd podczas nagrywania audio.')
      }

      mediaRecorder.onstop = () => {
        const recordedBlob = new Blob(chunksRef.current, {
          type: mediaRecorder.mimeType || 'audio/webm',
        })

        chunksRef.current = []
        mediaRecorderRef.current = null
        isStoppingRef.current = false
        cleanupStream()

        void Promise.resolve(onRecordingCompleteRef.current(recordedBlob)).catch((error) => {
          onErrorRef.current?.(toMessage(error, 'Nie udało się obsłużyć wyniku nagrania.'))
        })
      }

      const audioContextCtor =
        window.AudioContext ?? (window as WindowWithWebkitAudioContext).webkitAudioContext
      if (audioContextCtor) {
        const audioContext = new audioContextCtor()
        const source = audioContext.createMediaStreamSource(stream)
        const analyser = audioContext.createAnalyser()
        analyser.fftSize = 2048
        source.connect(analyser)

        audioContextRef.current = audioContext
        analyserRef.current = analyser
        analyserDataRef.current = new Uint8Array(new ArrayBuffer(analyser.fftSize))
        monitorSilenceRef.current = () => {
          const currentAnalyser = analyserRef.current
          const currentData = analyserDataRef.current
          if (!currentAnalyser || !currentData || !isRecordingRef.current) {
            return
          }

          currentAnalyser.getByteTimeDomainData(currentData)
          let squareSum = 0
          for (let index = 0; index < currentData.length; index += 1) {
            const normalizedValue = (currentData[index] - 128) / 128
            squareSum += normalizedValue * normalizedValue
          }

          const rms = Math.sqrt(squareSum / currentData.length)
          const now = performance.now()
          if (rms < silenceThreshold) {
            if (silenceStartedAtRef.current === null) {
              silenceStartedAtRef.current = now
            }
            if (now - silenceStartedAtRef.current >= silenceDurationMs) {
              stopRecording()
              return
            }
          } else {
            silenceStartedAtRef.current = null
          }

          const monitorSilence = monitorSilenceRef.current
          if (monitorSilence) {
            rafIdRef.current = window.requestAnimationFrame(monitorSilence)
          }
        }
      }

      mediaRecorder.start(CHUNK_DURATION_MS)
      isRecordingRef.current = true
      setIsRecording(true)
      if (monitorSilenceRef.current) {
        rafIdRef.current = window.requestAnimationFrame(monitorSilenceRef.current)
      }
    } catch (error) {
      cleanupStream()
      if (stream) {
        stream.getTracks().forEach((track) => {
          track.stop()
        })
      }
      throw new Error(toMessage(error, 'Nie udało się rozpocząć nagrywania audio.'))
    }
  }, [cleanupStream, silenceDurationMs, silenceThreshold, stopRecording])

  useEffect(() => {
    return () => {
      stopRecording()
      cleanupStream()
    }
  }, [cleanupStream, stopRecording])

  return {
    isRecording,
    startRecording,
    stopRecording,
  }
}
