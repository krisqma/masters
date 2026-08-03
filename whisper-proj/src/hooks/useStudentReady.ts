import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchHealth,
  type StudentStatus,
} from '../services/healthService'

const POLL_WARMING_MS = 2_000
const POLL_WARM_MS = 30_000

export interface StudentReadyState {
  status: StudentStatus
  error: string | null
  backendReachable: boolean
  studentReady: boolean
  markWarming: () => void
  refreshNow: () => void
}

const toMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Nie udało się sprawdzić stanu modelu.'

export function useStudentReady(): StudentReadyState {
  const [status, setStatus] = useState<StudentStatus>('warming')
  const [error, setError] = useState<string | null>(null)
  const [backendReachable, setBackendReachable] = useState(false)
  const statusRef = useRef<StudentStatus>(status)
  const pollRef = useRef<(() => Promise<void>) | null>(null)

  useEffect(() => {
    statusRef.current = status
  }, [status])

  const markWarming = useCallback(() => {
    setStatus('warming')
    setError(null)
  }, [])

  const refreshNow = useCallback(() => {
    void pollRef.current?.()
  }, [])

  useEffect(() => {
    let cancelled = false
    let timerId: number | null = null

    const scheduleNext = () => {
      if (cancelled) {
        return
      }
      const delay =
        statusRef.current === 'warm' ? POLL_WARM_MS : POLL_WARMING_MS
      timerId = window.setTimeout(() => {
        void poll()
      }, delay)
    }

    const poll = async () => {
      try {
        const health = await fetchHealth()
        if (cancelled) {
          return
        }
        setBackendReachable(true)
        setStatus(health.student_status)
        setError(
          health.student_status === 'error'
            ? health.student_error || 'Keep-warm modelu nie powiódł się.'
            : null,
        )
      } catch (pollError) {
        if (cancelled) {
          return
        }
        setBackendReachable(false)
        setStatus('error')
        setError(toMessage(pollError))
      } finally {
        scheduleNext()
      }
    }

    pollRef.current = poll
    void poll()

    return () => {
      cancelled = true
      pollRef.current = null
      if (timerId !== null) {
        window.clearTimeout(timerId)
      }
    }
  }, [])

  return {
    status,
    error,
    backendReachable,
    studentReady: status === 'warm',
    markWarming,
    refreshNow,
  }
}
