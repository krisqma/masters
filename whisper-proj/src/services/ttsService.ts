const DEFAULT_API_URL = 'http://127.0.0.1:8000'

const normalizeBaseUrl = (rawUrl?: string) => {
  if (!rawUrl) {
    return DEFAULT_API_URL
  }
  return rawUrl.replace(/\/+$/, '')
}

const resolveApiBaseUrl = () => normalizeBaseUrl(import.meta.env.VITE_API_URL)

interface TtsErrorResponse {
  detail?: string
}

const toMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Wystąpił nieoczekiwany błąd TTS.'

export async function synthesizeSpeech(
  text: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const trimmed = text.trim()
  if (!trimmed) {
    throw new Error('Brak tekstu do syntezy mowy.')
  }

  const response = await fetch(`${resolveApiBaseUrl()}/api/tts`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: trimmed }),
    signal,
  })

  if (!response.ok) {
    let detail = `TTS HTTP ${response.status}`
    try {
      const body = (await response.json()) as TtsErrorResponse
      if (typeof body.detail === 'string' && body.detail.trim()) {
        detail = body.detail
      }
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail)
  }

  return response.blob()
}

export async function playAudioBlob(blob: Blob, signal?: AbortSignal): Promise<void> {
  const objectUrl = URL.createObjectURL(blob)
  const audio = new Audio(objectUrl)

  const cleanup = () => {
    URL.revokeObjectURL(objectUrl)
    audio.onended = null
    audio.onerror = null
  }

  return new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      audio.pause()
      audio.removeAttribute('src')
      cleanup()
      reject(new DOMException('TTS playback aborted', 'AbortError'))
    }

    if (signal?.aborted) {
      cleanup()
      reject(new DOMException('TTS playback aborted', 'AbortError'))
      return
    }

    signal?.addEventListener('abort', onAbort, { once: true })

    audio.onended = () => {
      signal?.removeEventListener('abort', onAbort)
      cleanup()
      resolve()
    }
    audio.onerror = () => {
      signal?.removeEventListener('abort', onAbort)
      cleanup()
      reject(new Error('Nie udało się odtworzyć syntezowanego audio.'))
    }

    void audio.play().catch((error: unknown) => {
      signal?.removeEventListener('abort', onAbort)
      cleanup()
      reject(error instanceof Error ? error : new Error(toMessage(error)))
    })
  })
}
