import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

interface UseWakeWordOptions {
  enabled: boolean
  onWakeWordDetected: () => void
  /** Hard failures only (permission / unsupported). Soft network blips are handled internally. */
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
  'hej wilgus',
  'hej wilga',
  'hej wilgo',
  'hej wilko',
  'ej wilgus',
  'ej wilga',
  'ej wilgo',
  'ej wilko',
  'hey wilgus',
  'hey wilga',
]

/** Chrome often emits these during normal continuous listening — never spam the UI. */
const SOFT_ERRORS = new Set(['aborted', 'no-speech', 'network'])

const HARD_ERRORS = new Set(['not-allowed', 'service-not-allowed', 'audio-capture'])

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
  const restartTimerRef = useRef<number | null>(null)
  const networkFailStreakRef = useRef(0)
  const hardErrorReportedRef = useRef(false)

  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    wakeWordCallbackRef.current = onWakeWordDetected
  }, [onWakeWordDetected])

  useEffect(() => {
    errorCallbackRef.current = onError
  }, [onError])

  const clearRestartTimer = useCallback(() => {
    if (restartTimerRef.current !== null) {
      window.clearTimeout(restartTimerRef.current)
      restartTimerRef.current = null
    }
  }, [])

  const setListening = useCallback((value: boolean) => {
    isListeningRef.current = value
    setIsListening(value)
  }, [])

  const startRecognition = useCallback(() => {
    const recognition = recognitionRef.current
    if (!recognition || !enabledRef.current || isListeningRef.current) {
      return
    }

    shouldRestartRef.current = true
    try {
      recognition.start()
      setListening(true)
    } catch {
      // Browser may still be tearing down the previous session.
      setListening(false)
      clearRestartTimer()
      restartTimerRef.current = window.setTimeout(() => {
        restartTimerRef.current = null
        if (shouldRestartRef.current && enabledRef.current) {
          startRecognition()
        }
      }, 400)
    }
  }, [clearRestartTimer, setListening])

  const start = useCallback(() => {
    networkFailStreakRef.current = 0
    hardErrorReportedRef.current = false
    clearRestartTimer()
    startRecognition()
  }, [clearRestartTimer, startRecognition])

  const stop = useCallback(() => {
    clearRestartTimer()
    shouldRestartRef.current = false
    setListening(false)

    const recognition = recognitionRef.current
    if (!recognition) {
      return
    }

    try {
      recognition.stop()
    } catch {
      // Ignored intentionally - stop can throw when recognition has not started yet.
    }
  }, [clearRestartTimer, setListening])

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
      networkFailStreakRef.current = 0
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = event.results[index]?.[0]?.transcript ?? ''
        if (!transcript) {
          continue
        }

        if (hasWakeWord(transcript)) {
          shouldRestartRef.current = false
          clearRestartTimer()
          setListening(false)
          try {
            recognition.stop()
          } catch {
            // ignore
          }
          wakeWordCallbackRef.current()
          break
        }
      }
    }

    recognition.onerror = (event) => {
      const code = event.error ?? ''

      if (SOFT_ERRORS.has(code)) {
        if (code === 'network') {
          networkFailStreakRef.current += 1
          // After repeated network failures Chrome will just keep failing — slow down restarts.
          if (networkFailStreakRef.current >= 8 && !hardErrorReportedRef.current) {
            hardErrorReportedRef.current = true
            errorCallbackRef.current?.(
              'Wake word: Chrome Speech Recognition traci sieć. Sprawdź internet albo użyj przycisku nagrywania.',
            )
          }
        }
        return
      }

      if (HARD_ERRORS.has(code)) {
        shouldRestartRef.current = false
        clearRestartTimer()
        setListening(false)
        if (!hardErrorReportedRef.current) {
          hardErrorReportedRef.current = true
          errorCallbackRef.current?.(
            code === 'not-allowed' || code === 'service-not-allowed'
              ? 'Brak zgody na mikrofon dla wake word. Użyj przycisku albo odblokuj uprawnienia.'
              : `Błąd nasłuchiwania (${code}).`,
          )
        }
        return
      }

      // Unknown errors: report once, keep trying with backoff via onend.
      if (!hardErrorReportedRef.current) {
        hardErrorReportedRef.current = true
        errorCallbackRef.current?.(`Błąd nasłuchiwania (${code || 'unknown'}).`)
      }
    }

    recognition.onend = () => {
      setListening(false)
      if (!shouldRestartRef.current || !enabledRef.current) {
        return
      }

      const streak = networkFailStreakRef.current
      // Backoff: 300ms → ~5s cap so we don't hammer Chrome/Google on network blips.
      const delayMs = Math.min(5000, 300 * 2 ** Math.min(streak, 4))
      clearRestartTimer()
      restartTimerRef.current = window.setTimeout(() => {
        restartTimerRef.current = null
        if (shouldRestartRef.current && enabledRef.current) {
          startRecognition()
        }
      }, delayMs)
    }

    recognitionRef.current = recognition

    return () => {
      shouldRestartRef.current = false
      clearRestartTimer()
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
  }, [clearRestartTimer, recognitionCtor, setListening, startRecognition])

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
