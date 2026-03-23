import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

interface UseWakeWordOptions {
  enabled: boolean
  onWakeWordDetected: () => void
  onError?: (message: string) => void
}

interface UseWakeWordResult {
  isSupported: boolean
  isListening: boolean
  start: () => void
  stop: () => void
}

type SpeechRecognitionResultLike = {
  0?: {
    transcript?: string
  }
}

type SpeechRecognitionEventLike = {
  resultIndex: number
  results: ArrayLike<SpeechRecognitionResultLike>
}

type SpeechRecognitionErrorEventLike = {
  error?: string
  message?: string
}

type SpeechRecognitionInstance = {
  lang: string
  continuous: boolean
  interimResults: boolean
  maxAlternatives: number
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionCtor = new () => SpeechRecognitionInstance

type WindowWithSpeechRecognition = Window & {
  SpeechRecognition?: SpeechRecognitionCtor
  webkitSpeechRecognition?: SpeechRecognitionCtor
}

const WAKE_WORD_PATTERNS = [
  'hej wilga',
  'hej wilgo',
  'hej wilko',
  'ej wilga',
  'ej wilgo',
  'ej wilko',
]

const normalizeTranscript = (value: string) =>
  value
    .toLocaleLowerCase('pl-PL')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const hasWakeWord = (value: string) => {
  const normalized = normalizeTranscript(value)
  if (WAKE_WORD_PATTERNS.some((pattern) => normalized.includes(pattern))) {
    return true
  }

  return /(?:h|e)j\s+wil(g|k)\w*/.test(normalized)
}

export function useWakeWord({
  enabled,
  onWakeWordDetected,
  onError,
}: UseWakeWordOptions): UseWakeWordResult {
  const [isListening, setIsListening] = useState(false)

  const recognitionCtor = useMemo(() => {
    if (typeof window === 'undefined') {
      return null
    }
    const recognitionWindow = window as WindowWithSpeechRecognition
    return (
      recognitionWindow.SpeechRecognition ?? recognitionWindow.webkitSpeechRecognition ?? null
    )
  }, [])
  const isSupported = recognitionCtor !== null

  const isListeningRef = useRef(false)
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null)
  const shouldRestartRef = useRef(false)
  const enabledRef = useRef(enabled)
  const wakeWordCallbackRef = useRef(onWakeWordDetected)
  const errorCallbackRef = useRef(onError)

  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    wakeWordCallbackRef.current = onWakeWordDetected
  }, [onWakeWordDetected])

  useEffect(() => {
    errorCallbackRef.current = onError
  }, [onError])

  const setListening = useCallback((value: boolean) => {
    isListeningRef.current = value
    setIsListening(value)
  }, [])

  const start = useCallback(() => {
    const recognition = recognitionRef.current
    if (!recognition || isListeningRef.current) {
      return
    }

    shouldRestartRef.current = true

    try {
      recognition.start()
      setListening(true)
    } catch (error) {
      shouldRestartRef.current = false
      setListening(false)
      const message =
        error instanceof Error
          ? error.message
          : 'Nie udało się uruchomić nasłuchiwania wake word.'
      errorCallbackRef.current?.(message)
    }
  }, [setListening])

  const stop = useCallback(() => {
    const recognition = recognitionRef.current
    if (!recognition) {
      return
    }

    shouldRestartRef.current = false
    setListening(false)

    try {
      recognition.stop()
    } catch {
      // Ignored intentionally - stop can throw when recognition has not started yet.
    }
  }, [setListening])

  useEffect(() => {
    if (!recognitionCtor) {
      return
    }

    const recognition = new recognitionCtor()
    recognition.lang = 'pl-PL'
    recognition.continuous = true
    recognition.interimResults = true
    recognition.maxAlternatives = 1

    recognition.onresult = (event) => {
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = event.results[index]?.[0]?.transcript ?? ''
        if (!transcript) {
          continue
        }

        if (hasWakeWord(transcript)) {
          shouldRestartRef.current = false
          setListening(false)
          recognition.stop()
          wakeWordCallbackRef.current()
          break
        }
      }
    }

    recognition.onerror = (event) => {
      if (event.error === 'aborted') {
        return
      }

      const message = event.error
        ? `Błąd nasłuchiwania (${event.error}). ${event.message ?? ''}`.trim()
        : 'Wystąpił błąd nasłuchiwania.'
      errorCallbackRef.current?.(message)
    }

    recognition.onend = () => {
      setListening(false)
      if (!shouldRestartRef.current || !enabledRef.current) {
        return
      }

      try {
        recognition.start()
        setListening(true)
      } catch {
        // Ignored intentionally - the browser may still release microphone resources.
      }
    }

    recognitionRef.current = recognition

    return () => {
      shouldRestartRef.current = false
      recognition.onresult = null
      recognition.onerror = null
      recognition.onend = null
      try {
        recognition.stop()
      } catch {
        // Ignored intentionally.
      }
      recognitionRef.current = null
      setListening(false)
    }
  }, [recognitionCtor, setListening])

  useEffect(() => {
    if (!isSupported) {
      return
    }

    const timerId = window.setTimeout(() => {
      if (enabled) {
        start()
      } else {
        stop()
      }
    }, 0)

    return () => {
      window.clearTimeout(timerId)
    }
  }, [enabled, isSupported, start, stop])

  return useMemo(
    () => ({
      isSupported,
      isListening,
      start,
      stop,
    }),
    [isListening, isSupported, start, stop],
  )
}
