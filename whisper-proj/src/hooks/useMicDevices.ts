import { useCallback, useEffect, useState } from 'react'
import {
  listAudioInputDevices,
  persistAudioInputDeviceId,
  pickPreferredDeviceId,
  type AudioInputDevice,
} from '../utils/audioDevices'

export function useMicDevices() {
  const [devices, setDevices] = useState<AudioInputDevice[]>([])
  const [deviceId, setDeviceIdState] = useState('')
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const next = await listAudioInputDevices()
      setDevices(next)
      setDeviceIdState((current) => {
        if (current && next.some((device) => device.deviceId === current)) {
          return current
        }
        return pickPreferredDeviceId(next)
      })
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się listować mikrofonów.')
    }
  }, [])

  useEffect(() => {
    void refresh()
    const onChange = () => {
      void refresh()
    }
    navigator.mediaDevices?.addEventListener?.('devicechange', onChange)
    return () => {
      navigator.mediaDevices?.removeEventListener?.('devicechange', onChange)
    }
  }, [refresh])

  const setDeviceId = useCallback((nextId: string) => {
    setDeviceIdState(nextId)
    persistAudioInputDeviceId(nextId)
  }, [])

  return {
    devices,
    deviceId,
    setDeviceId,
    refresh,
    error,
  }
}
