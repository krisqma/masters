import { useCallback, useEffect, useRef, useState } from 'react'
import { buildAudioConstraints } from '../utils/audioDevices'

interface UseAudioRecorderOptions {
  onRecordingComplete: (audioBlob: Blob) => void | Promise<void>
  onError?: (message: string) => void
  deviceId?: string
  silenceDurationMs?: number
  silenceThreshold?: number
}

interface UseAudioRecorderResult {
  isRecording: boolean
  inputLevel: number
  activeTrackLabel: string
  startRecording: () => Promise<void>
  stopRecording: () => void
}

const DEFAULT_SILENCE_DURATION_MS = 1_600
const DEFAULT_SILENCE_THRESHOLD = 0.018
const MIN_RECORDING_MS = 2_500
const MAX_RECORDING_MS = 20_000
const MIN_BLOB_BYTES = 2_500
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
  deviceId = '',
  silenceDurationMs = DEFAULT_SILENCE_DURATION_MS,
  silenceThreshold = DEFAULT_SILENCE_THRESHOLD,
}: UseAudioRecorderOptions): UseAudioRecorderResult {
  const [isRecording, setIsRecording] = useState(false)
  const [inputLevel, setInputLevel] = useState(0)
  const [activeTrackLabel, setActiveTrackLabel] = useState('')
  const isRecordingRef = useRef(false)
  const deviceIdRef = useRef(deviceId)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const monitorStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const analyserDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null)
  const monitorSilenceRef = useRef<(() => void) | null>(null)
  const silenceStartedAtRef = useRef<number | null>(null)
  const recordingStartedAtRef = useRef<number | null>(null)
  const heardSpeechRef = useRef(false)
  const maxTimerRef = useRef<number | null>(null)
  const rafIdRef = useRef<number | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const isStoppingRef = useRef(false)
  const onRecordingCompleteRef = useRef(onRecordingComplete)
  const onErrorRef = useRef(onError)

  useEffect(() => {
    deviceIdRef.current = deviceId
  }, [deviceId])

  useEffect(() => {
    onRecordingCompleteRef.current = onRecordingComplete
  }, [onRecordingComplete])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  const clearMaxTimer = useCallback(() => {
    if (maxTimerRef.current !== null) {
      window.clearTimeout(maxTimerRef.current)
      maxTimerRef.current = null
    }
  }, [])

  const cleanupStream = useCallback(() => {
    clearMaxTimer()
    if (rafIdRef.current !== null) {
      window.cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }

    silenceStartedAtRef.current = null
    recordingStartedAtRef.current = null
    heardSpeechRef.current = false
    analyserDataRef.current = null
    monitorSilenceRef.current = null
    analyserRef.current?.disconnect()
    analyserRef.current = null

    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => undefined)
      audioContextRef.current = null
    }

    if (monitorStreamRef.current) {
      monitorStreamRef.current.getTracks().forEach((track) => track.stop())
      monitorStreamRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    setInputLevel(0)
  }, [clearMaxTimer])

  const stopRecording = useCallback(() => {
    const mediaRecorder = mediaRecorderRef.current
    if (!mediaRecorder || mediaRecorder.state === 'inactive' || isStoppingRef.current) {
      return
    }

    isStoppingRef.current = true
    clearMaxTimer()
    // Do NOT call requestData() before stop — it often yields a corrupt/partial webm
    // that ffmpeg rejects with "Invalid data found when processing input".
    mediaRecorder.stop()
    isRecordingRef.current = false
    setIsRecording(false)
  }, [clearMaxTimer])

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
    recordingStartedAtRef.current = null
    heardSpeechRef.current = false
    silenceStartedAtRef.current = null
    clearMaxTimer()

    let stream: MediaStream | null = null
    try {
      const constraints = buildAudioConstraints(deviceIdRef.current || undefined)
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: constraints })
      } catch (error) {
        // Exact deviceId can fail if device unplugged — fall back to default.
        if (deviceIdRef.current) {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: buildAudioConstraints(),
          })
        } else {
          throw error
        }
      }
      streamRef.current = stream

      const track = stream.getAudioTracks()[0]
      if (!track || track.readyState !== 'live') {
        throw new Error('Mikrofon nie jest aktywny. Wybierz inne urządzenie wejściowe.')
      }
      track.enabled = true
      const label = track.label || 'Nieznany mikrofon'
      setActiveTrackLabel(label)
      console.info('[Wilga] nagrywanie z mikrofonu:', label, track.getSettings())

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

        const chunkCount = chunksRef.current.length
        chunksRef.current = []
        mediaRecorderRef.current = null
        isStoppingRef.current = false
        cleanupStream()

        if (recordedBlob.size < MIN_BLOB_BYTES) {
          onErrorRef.current?.(
            `Nagranie puste (blob=${recordedBlob.size} B, chunks=${chunkCount}). ` +
              `Sprawdź wybrany mikrofon w UI (teraz: „${label}”). ` +
              'Chrome czasem bierze Continuity/iPhone zamiast Mic MacBooka.',
          )
          return
        }

        void Promise.resolve(onRecordingCompleteRef.current(recordedBlob)).catch((error) => {
          onErrorRef.current?.(toMessage(error, 'Nie udało się obsłużyć wyniku nagrania.'))
        })
      }

      const audioContextCtor =
        window.AudioContext ?? (window as WindowWithWebkitAudioContext).webkitAudioContext
      if (audioContextCtor) {
        const audioContext = new audioContextCtor()
        const monitorStream = stream.clone()
        monitorStreamRef.current = monitorStream
        const source = audioContext.createMediaStreamSource(monitorStream)
        const analyser = audioContext.createAnalyser()
        analyser.fftSize = 2048
        source.connect(analyser)
        void audioContext.resume()

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
          setInputLevel(Math.min(1, rms * 4))

          const now = performance.now()
          const startedAt = recordingStartedAtRef.current
          const minRecordingReached =
            startedAt !== null && now - startedAt >= MIN_RECORDING_MS

          if (rms >= silenceThreshold) {
            heardSpeechRef.current = true
            silenceStartedAtRef.current = null
          } else if (heardSpeechRef.current) {
            if (silenceStartedAtRef.current === null) {
              silenceStartedAtRef.current = now
            }
            if (
              minRecordingReached &&
              now - silenceStartedAtRef.current >= silenceDurationMs
            ) {
              stopRecording()
              return
            }
          }

          const monitorSilence = monitorSilenceRef.current
          if (monitorSilence) {
            rafIdRef.current = window.requestAnimationFrame(monitorSilence)
          }
        }
      }

      // One complete container on stop (timeslices frequently break ffmpeg decode).
      mediaRecorder.start()
      recordingStartedAtRef.current = performance.now()
      isRecordingRef.current = true
      setIsRecording(true)
      maxTimerRef.current = window.setTimeout(() => {
        stopRecording()
      }, MAX_RECORDING_MS)
      if (monitorSilenceRef.current) {
        rafIdRef.current = window.requestAnimationFrame(monitorSilenceRef.current)
      }
    } catch (error) {
      cleanupStream()
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
      throw new Error(toMessage(error, 'Nie udało się rozpocząć nagrywania audio.'))
    }
  }, [cleanupStream, clearMaxTimer, silenceDurationMs, silenceThreshold, stopRecording])

  useEffect(() => {
    return () => {
      isStoppingRef.current = true
      clearMaxTimer()
      const mediaRecorder = mediaRecorderRef.current
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.onstop = null
        try {
          mediaRecorder.stop()
        } catch {
          // ignore
        }
      }
      mediaRecorderRef.current = null
      isRecordingRef.current = false
      cleanupStream()
    }
  }, [cleanupStream, clearMaxTimer])

  return {
    isRecording,
    inputLevel,
    activeTrackLabel,
    startRecording,
    stopRecording,
  }
}
