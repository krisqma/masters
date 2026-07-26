import { blobToWav } from '../utils/encodeWav'

interface WhisperTranscriptionResponse {
  text?: string
  detail?: string
}

const DEFAULT_API_URL = 'http://127.0.0.1:8000'

const normalizeBaseUrl = (rawUrl?: string) => {
  if (!rawUrl) {
    return DEFAULT_API_URL
  }
  return rawUrl.replace(/\/+$/, '')
}

const resolveApiBaseUrl = () => normalizeBaseUrl(import.meta.env.VITE_API_URL)

export async function transcribeAudio(audioBlob: Blob): Promise<string> {
  const apiBaseUrl = resolveApiBaseUrl()

  // Browser MediaRecorder webm is often unreadable by ffmpeg/Whisper ("Invalid data").
  // Re-encode to PCM WAV in the browser first.
  let uploadBlob: Blob
  let filename: string
  try {
    uploadBlob = await blobToWav(audioBlob)
    filename = `recording-${Date.now()}.wav`
  } catch (error) {
    console.warn('[Wilga] WAV convert failed, falling back to raw blob', error)
    uploadBlob = audioBlob
    const mimeType = audioBlob.type || 'audio/webm'
    const extension = mimeType.includes('mp4') ? 'm4a' : 'webm'
    filename = `recording-${Date.now()}.${extension}`
  }

  const file = new File([uploadBlob], filename, {
    type: uploadBlob.type || 'audio/wav',
  })

  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${apiBaseUrl}/api/transcribe`, {
    method: 'POST',
    body: formData,
  })

  let payload: WhisperTranscriptionResponse | null = null
  try {
    payload = (await response.json()) as WhisperTranscriptionResponse
  } catch {
    payload = null
  }

  if (!response.ok) {
    const apiMessage = payload?.detail
    throw new Error(apiMessage || `Błąd transkrypcji (HTTP ${response.status}).`)
  }

  const transcript = payload?.text?.trim()
  if (!transcript) {
    throw new Error('API Whisper zwróciło pustą transkrypcję.')
  }

  return transcript
}
