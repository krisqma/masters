import { useEffect, useRef } from 'react'
import { buildAudioConstraints } from '../utils/audioDevices'

export const FOLLOW_UP_WINDOW_MS = 30_000
const SPEECH_THRESHOLD = 0.025
const SPEECH_HOLD_MS = 300

interface UseFollowUpListenerOptions {
  enabled: boolean
  deviceId?: string
  onSpeechDetected: () => void
  onTimeout: () => void
  /** Remaining seconds updates for UI (optional). */
  onTick?: (secondsLeft: number) => void
}

type WindowWithWebkitAudioContext = Window & {
  webkitAudioContext?: typeof AudioContext
}

/**
 * Hot mic after Wilga finishes speaking: wait for energy above threshold,
 * then hand off to MediaRecorder. Times out back to wake-word mode.
 */
export function useFollowUpListener({
  enabled,
  deviceId = '',
  onSpeechDetected,
  onTimeout,
  onTick,
}: UseFollowUpListenerOptions): void {
  const onSpeechDetectedRef = useRef(onSpeechDetected)
  const onTimeoutRef = useRef(onTimeout)
  const onTickRef = useRef(onTick)
  const deviceIdRef = useRef(deviceId)

  useEffect(() => {
    onSpeechDetectedRef.current = onSpeechDetected
  }, [onSpeechDetected])

  useEffect(() => {
    onTimeoutRef.current = onTimeout
  }, [onTimeout])

  useEffect(() => {
    onTickRef.current = onTick
  }, [onTick])

  useEffect(() => {
    deviceIdRef.current = deviceId
  }, [deviceId])

  useEffect(() => {
    if (!enabled) {
      return
    }
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      onTimeoutRef.current()
      return
    }

    let cancelled = false
    let fired = false
    let stream: MediaStream | null = null
    let audioContext: AudioContext | null = null
    let analyser: AnalyserNode | null = null
    let data: Uint8Array<ArrayBuffer> | null = null
    let rafId: number | null = null
    let speechStartedAt: number | null = null
    let timeoutId: number | null = null
    let tickId: number | null = null
    const startedAt = performance.now()

    const cleanup = () => {
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId)
        rafId = null
      }
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
        timeoutId = null
      }
      if (tickId !== null) {
        window.clearInterval(tickId)
        tickId = null
      }
      if (audioContext) {
        void audioContext.close().catch(() => undefined)
        audioContext = null
      }
      analyser = null
      data = null
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
        stream = null
      }
    }

    const fireSpeech = () => {
      if (cancelled || fired) {
        return
      }
      fired = true
      cleanup()
      onSpeechDetectedRef.current()
    }

    const monitor = () => {
      if (cancelled || fired || !analyser || !data) {
        return
      }

      analyser.getByteTimeDomainData(data)
      let sumSquares = 0
      for (let index = 0; index < data.length; index += 1) {
        const centered = ((data[index] ?? 128) - 128) / 128
        sumSquares += centered * centered
      }
      const rms = Math.sqrt(sumSquares / data.length)

      if (rms >= SPEECH_THRESHOLD) {
        if (speechStartedAt === null) {
          speechStartedAt = performance.now()
        } else if (performance.now() - speechStartedAt >= SPEECH_HOLD_MS) {
          fireSpeech()
          return
        }
      } else {
        speechStartedAt = null
      }

      rafId = window.requestAnimationFrame(monitor)
    }

    const start = async () => {
      try {
        const constraints = buildAudioConstraints(deviceIdRef.current || undefined)
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: constraints })
        } catch {
          if (deviceIdRef.current) {
            stream = await navigator.mediaDevices.getUserMedia({
              audio: buildAudioConstraints(),
            })
          } else {
            throw new Error('Brak dostępu do mikrofonu w trybie follow-up.')
          }
        }

        if (cancelled) {
          cleanup()
          return
        }

        const audioContextCtor =
          window.AudioContext ?? (window as WindowWithWebkitAudioContext).webkitAudioContext
        if (!audioContextCtor) {
          onTimeoutRef.current()
          cleanup()
          return
        }

        audioContext = new audioContextCtor()
        const source = audioContext.createMediaStreamSource(stream)
        analyser = audioContext.createAnalyser()
        analyser.fftSize = 2048
        source.connect(analyser)
        void audioContext.resume()
        data = new Uint8Array(new ArrayBuffer(analyser.fftSize))

        onTickRef.current?.(Math.ceil(FOLLOW_UP_WINDOW_MS / 1000))
        tickId = window.setInterval(() => {
          const leftMs = FOLLOW_UP_WINDOW_MS - (performance.now() - startedAt)
          onTickRef.current?.(Math.max(0, Math.ceil(leftMs / 1000)))
        }, 250)

        timeoutId = window.setTimeout(() => {
          if (cancelled || fired) {
            return
          }
          fired = true
          cleanup()
          onTimeoutRef.current()
        }, FOLLOW_UP_WINDOW_MS)

        rafId = window.requestAnimationFrame(monitor)
      } catch {
        if (!cancelled) {
          onTimeoutRef.current()
        }
        cleanup()
      }
    }

    void start()

    return () => {
      cancelled = true
      cleanup()
    }
  }, [enabled, deviceId])
}
