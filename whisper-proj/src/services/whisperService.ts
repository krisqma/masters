interface WhisperTranscriptionResponse {
  text?: string
  error?: {
    message?: string
  }
}

const OPENAI_TRANSCRIPTION_ENDPOINT = 'https://api.openai.com/v1/audio/transcriptions'
const DEFAULT_API_URL = 'http://127.0.0.1:8000'

const normalizeBaseUrl = (rawUrl?: string) => {
  if (!rawUrl) {
    return DEFAULT_API_URL
  }
  return rawUrl.replace(/\/+$/, '')
}

const resolveApiBaseUrl = () => normalizeBaseUrl(import.meta.env.VITE_API_URL)

const guessFileExtension = (mimeType: string) => {
  if (mimeType.includes('mpeg') || mimeType.includes('mp3')) {
    return 'mp3'
  }
  if (mimeType.includes('mp4')) {
    return 'm4a'
  }
  if (mimeType.includes('wav')) {
    return 'wav'
  }

  return 'webm'
}

export async function transcribeAudio(audioBlob: Blob): Promise<string> {
  const proxyUrl = import.meta.env.VITE_WHISPER_PROXY_URL?.trim()
  const apiKey = import.meta.env.VITE_OPENAI_API_KEY?.trim()
  const apiBaseUrl = resolveApiBaseUrl()
  const mimeType = audioBlob.type || 'audio/webm'
  const extension = guessFileExtension(mimeType)
  const file = new File([audioBlob], `recording-${Date.now()}.${extension}`, {
    type: mimeType,
  })

  const formData = new FormData()
  formData.append('file', file)
  if (!proxyUrl && apiKey) {
    formData.append('model', 'whisper-1')
    formData.append('language', 'pl')
    formData.append('response_format', 'json')
  }

  const requestUrl = proxyUrl || (apiKey ? OPENAI_TRANSCRIPTION_ENDPOINT : `${apiBaseUrl}/api/transcribe`)
  const headers: HeadersInit = {}
  if (requestUrl === OPENAI_TRANSCRIPTION_ENDPOINT) {
    if (!apiKey) {
      throw new Error('Brak VITE_OPENAI_API_KEY. Uzupełnij klucz w pliku .env.local.')
    }
    headers.Authorization = `Bearer ${apiKey}`
  }

  const response = await fetch(requestUrl, {
    method: 'POST',
    headers,
    body: formData,
  })

  let payload: WhisperTranscriptionResponse | null = null
  try {
    payload = (await response.json()) as WhisperTranscriptionResponse
  } catch {
    payload = null
  }

  if (!response.ok) {
    const apiMessage = payload?.error?.message
    throw new Error(apiMessage || `Błąd transkrypcji (HTTP ${response.status}).`)
  }

  const transcript = payload?.text?.trim()
  if (!transcript) {
    throw new Error('API Whisper zwróciło pustą transkrypcję.')
  }

  return transcript
}
