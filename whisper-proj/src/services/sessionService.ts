const DEFAULT_API_URL = 'http://127.0.0.1:8000'

const normalizeBaseUrl = (rawUrl?: string) => {
  if (!rawUrl) {
    return DEFAULT_API_URL
  }
  return rawUrl.replace(/\/+$/, '')
}

const resolveApiBaseUrl = () => normalizeBaseUrl(import.meta.env.VITE_API_URL)

export interface SessionNewResult {
  status: 'warming' | 'warm' | 'error'
  advanced: boolean
  source_timestamp: string | null
  published_facts_chars: number | null
  student_status: 'warming' | 'warm' | 'error'
  idle_rotate_seconds: number
}

export async function startNewSession(): Promise<SessionNewResult> {
  const response = await fetch(`${resolveApiBaseUrl()}/api/session/new`, {
    method: 'POST',
  })
  if (!response.ok) {
    throw new Error(`Nie udało się rozpocząć nowej sesji (HTTP ${response.status}).`)
  }
  const body = (await response.json()) as SessionNewResult
  return body
}
