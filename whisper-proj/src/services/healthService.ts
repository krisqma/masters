const DEFAULT_API_URL = 'http://127.0.0.1:8000'

const normalizeBaseUrl = (rawUrl?: string) => {
  if (!rawUrl) {
    return DEFAULT_API_URL
  }
  return rawUrl.replace(/\/+$/, '')
}

const resolveApiBaseUrl = () => normalizeBaseUrl(import.meta.env.VITE_API_URL)

export type StudentStatus = 'warming' | 'warm' | 'error'

export interface HealthResponse {
  status: string
  model: string
  device: string
  language: string
  student_warm: boolean
  student_status: StudentStatus
  student_error: string | null
}

export const fetchHealth = async (): Promise<HealthResponse> => {
  const response = await fetch(`${resolveApiBaseUrl()}/api/health`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(`Health check failed (${response.status}).`)
  }

  const body = (await response.json()) as Partial<HealthResponse>
  const studentStatus =
    body.student_status === 'warm' ||
    body.student_status === 'warming' ||
    body.student_status === 'error'
      ? body.student_status
      : body.student_warm
        ? 'warm'
        : 'warming'

  return {
    status: typeof body.status === 'string' ? body.status : 'unknown',
    model: typeof body.model === 'string' ? body.model : '',
    device: typeof body.device === 'string' ? body.device : '',
    language: typeof body.language === 'string' ? body.language : '',
    student_warm: Boolean(body.student_warm) || studentStatus === 'warm',
    student_status: studentStatus,
    student_error:
      typeof body.student_error === 'string' && body.student_error.trim()
        ? body.student_error
        : null,
  }
}
