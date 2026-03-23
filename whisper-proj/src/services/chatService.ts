const DEFAULT_API_URL = 'http://127.0.0.1:8000'

const normalizeBaseUrl = (rawUrl?: string) => {
  if (!rawUrl) {
    return DEFAULT_API_URL
  }
  return rawUrl.replace(/\/+$/, '')
}

const resolveApiBaseUrl = () => normalizeBaseUrl(import.meta.env.VITE_API_URL)

interface StreamChatReplyOptions {
  onChunk: (chunk: string) => void
}

interface ChatErrorResponse {
  detail?: string
}

export async function streamChatReply(
  message: string,
  { onChunk }: StreamChatReplyOptions,
): Promise<void> {
  const requestMessage = message.trim()
  if (!requestMessage) {
    throw new Error('Nie można wysłać pustej wiadomości do modelu.')
  }

  const response = await fetch(`${resolveApiBaseUrl()}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message: requestMessage }),
  })

  if (!response.ok) {
    let payload: ChatErrorResponse | null = null
    try {
      payload = (await response.json()) as ChatErrorResponse
    } catch {
      payload = null
    }
    throw new Error(payload?.detail || `Błąd czatu (HTTP ${response.status}).`)
  }

  if (!response.body) {
    throw new Error('Serwer nie zwrócił strumienia odpowiedzi.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }

    const chunk = decoder.decode(value, { stream: true })
    if (chunk) {
      onChunk(chunk)
    }
  }

  const finalChunk = decoder.decode()
  if (finalChunk) {
    onChunk(finalChunk)
  }
}
