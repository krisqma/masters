const DEFAULT_API_URL = 'http://127.0.0.1:8000'

const normalizeBaseUrl = (rawUrl?: string) => {
  if (!rawUrl) {
    return DEFAULT_API_URL
  }
  return rawUrl.replace(/\/+$/, '')
}

const resolveApiBaseUrl = () => normalizeBaseUrl(import.meta.env.VITE_API_URL)
const CHAT_REQUEST_TIMEOUT_MS = 90_000

/** Keep in sync with backend CHAT_HISTORY_MAX_MESSAGES default. */
export const CHAT_HISTORY_MAX_MESSAGES = 6

export type ChatHistoryRole = 'user' | 'assistant'

export interface ChatHistoryItem {
  role: ChatHistoryRole
  content: string
}

type ChatStreamEvent =
  | {
      type: 'chunk'
      content: string
    }
  | {
      type: 'error'
      detail: string
    }
  | {
      type: 'done'
    }

interface StreamChatReplyOptions {
  history?: ChatHistoryItem[]
  onChunk: (chunk: string) => void
  onError?: (message: string) => void
  signal?: AbortSignal
  timeoutMs?: number
}

interface ChatErrorResponse {
  detail?: string
}

const toMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Wystąpił nieoczekiwany błąd czatu.'

const drainDecodedBuffer = (
  input: string,
  processLine: (line: string) => void,
  flush: boolean,
) => {
  const lines = input.split('\n')
  const remainder = flush ? '' : (lines.pop() ?? '')

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (line) {
      processLine(line)
    }
  }

  return remainder
}

export async function streamChatReply(
  message: string,
  {
    history = [],
    onChunk,
    onError,
    signal,
    timeoutMs = CHAT_REQUEST_TIMEOUT_MS,
  }: StreamChatReplyOptions,
): Promise<void> {
  const requestMessage = message.trim()
  if (!requestMessage) {
    throw new Error('Nie można wysłać pustej wiadomości do modelu.')
  }

  const trimmedHistory = history
    .map((item) => ({
      role: item.role,
      content: item.content.trim(),
    }))
    .filter((item) => item.content.length > 0)
    .slice(-CHAT_HISTORY_MAX_MESSAGES)

  const controller = new AbortController()
  let didTimeout = false
  const timeoutId = window.setTimeout(() => {
    didTimeout = true
    controller.abort()
  }, timeoutMs)

  const abortFromCaller = () => {
    controller.abort()
  }

  if (signal) {
    if (signal.aborted) {
      controller.abort()
    } else {
      signal.addEventListener('abort', abortFromCaller, { once: true })
    }
  }

  try {
    const response = await fetch(`${resolveApiBaseUrl()}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: requestMessage,
        history: trimmedHistory,
      }),
      signal: controller.signal,
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
    let buffer = ''
    let doneReceived = false
    let streamErrorReceived = false

    const processLine = (line: string) => {
      let event: ChatStreamEvent
      try {
        event = JSON.parse(line) as ChatStreamEvent
      } catch {
        throw new Error('Serwer zwrócił niepoprawny format strumienia odpowiedzi.')
      }

      if (event.type === 'chunk') {
        if (typeof event.content !== 'string' || !event.content) {
          return
        }
        onChunk(event.content)
        return
      }

      if (event.type === 'error') {
        streamErrorReceived = true
        onError?.(event.detail || 'Wystąpił błąd generowania odpowiedzi.')
        return
      }

      if (event.type === 'done') {
        doneReceived = true
        return
      }

      throw new Error('Serwer zwrócił nieznany typ zdarzenia strumienia.')
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      buffer = drainDecodedBuffer(buffer, processLine, false)
    }

    buffer += decoder.decode()
    buffer = drainDecodedBuffer(buffer, processLine, true)

    const trailingLine = buffer.trim()
    if (trailingLine) {
      processLine(trailingLine)
    }

    if (!doneReceived && !streamErrorReceived) {
      throw new Error('Strumień odpowiedzi został przerwany przed zakończeniem.')
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(
        didTimeout
          ? 'Generowanie odpowiedzi przekroczyło limit czasu.'
          : 'Żądanie odpowiedzi zostało przerwane.',
      )
    }
    throw new Error(toMessage(error))
  } finally {
    window.clearTimeout(timeoutId)
    signal?.removeEventListener('abort', abortFromCaller)
  }
}
