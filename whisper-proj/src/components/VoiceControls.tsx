import { useEffect, useId, useRef, useState } from 'react'
import type { AudioInputDevice } from '../utils/audioDevices'
import type { StudentStatus } from '../services/healthService'
import type { AppStatus } from '../types'

interface VoiceControlsProps {
  status: AppStatus
  isWakeWordSupported: boolean
  isWakeWordListening: boolean
  devices: AudioInputDevice[]
  deviceId: string
  onDeviceChange: (deviceId: string) => void
  onNewConversation: () => void
  followUpSecondsLeft?: number | null
  studentStatus: StudentStatus
  studentReady: boolean
  studentError: string | null
  backendReachable: boolean
}

function SettingsIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 8.25a3.75 3.75 0 1 0 0 7.5 3.75 3.75 0 0 0 0-7.5Z"
        stroke="currentColor"
        strokeWidth="1.7"
      />
      <path
        d="M19.43 12.98c.04-.32.07-.65.07-.98s-.03-.66-.07-.98l2.11-1.65a.5.5 0 0 0 .12-.64l-2-3.46a.5.5 0 0 0-.61-.22l-2.49 1a7.03 7.03 0 0 0-1.69-.98l-.38-2.65A.5.5 0 0 0 13.99 2h-4a.5.5 0 0 0-.5.42l-.38 2.65c-.61.25-1.18.58-1.69.98l-2.49-1a.5.5 0 0 0-.61.22l-2 3.46a.5.5 0 0 0 .12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65a.5.5 0 0 0-.12.64l2 3.46c.14.24.43.34.61.22l2.49-1c.51.4 1.08.73 1.69.98l.38 2.65a.5.5 0 0 0 .5.42h4a.5.5 0 0 0 .5-.42l.38-2.65c.61-.25 1.18-.58 1.69-.98l2.49 1c.24.1.48 0 .61-.22l2-3.46a.5.5 0 0 0-.12-.64l-2.11-1.65Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function VoiceControls({
  status,
  isWakeWordSupported,
  isWakeWordListening,
  devices,
  deviceId,
  onDeviceChange,
  onNewConversation,
  followUpSecondsLeft = null,
  studentStatus,
  studentReady,
  studentError,
  backendReachable,
}: VoiceControlsProps) {
  const [open, setOpen] = useState(false)
  const panelId = useId()
  const rootRef = useRef<HTMLDivElement>(null)
  const busy =
    status === 'RECORDING' || status === 'PROCESSING' || status === 'SPEAKING'

  useEffect(() => {
    if (!open) {
      return
    }

    const onPointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  let hint = ''
  if (status === 'LISTENING_WAKE_WORD') {
    hint = isWakeWordListening
      ? 'Nasłuch wake word aktywny'
      : isWakeWordSupported
        ? 'Uruchamiam wake word…'
        : 'Wake word niedostępny — użyj kliknięcia'
  } else if (status === 'LISTENING_FOLLOW_UP') {
    const seconds =
      typeof followUpSecondsLeft === 'number' ? followUpSecondsLeft : 30
    hint = `Follow-up: ${seconds} s`
  } else if (status === 'IDLE') {
    hint = 'Tryb ręczny — kliknij Wilgusia'
  }

  let readinessBanner: { className: string; text: string } | null = null
  if (!backendReachable || studentStatus === 'error') {
    readinessBanner = {
      className: 'model-readiness model-readiness-error',
      text: !backendReachable
        ? 'Nie mogę połączyć się z backendem — sprawdzam ponownie…'
        : studentError
          ? `Model na RPi niedostępny: ${studentError}`
          : 'Nie mogę połączyć się z modelem na RPi — sprawdzam ponownie…',
    }
  } else if (!studentReady) {
    readinessBanner = {
      className: 'model-readiness model-readiness-warming',
      text: 'Rozgrzewam model… zaraz będzie gotowy',
    }
  }

  return (
    <div className="settings-slot" ref={rootRef}>
      {readinessBanner ? (
        <p className={readinessBanner.className} role="status" aria-live="polite">
          {!studentReady && studentStatus !== 'error' && backendReachable ? (
            <span className="model-readiness-spinner" aria-hidden />
          ) : null}
          <span>{readinessBanner.text}</span>
        </p>
      ) : null}

      <button
        type="button"
        className={`settings-button ${open ? 'settings-button-open' : ''}`}
        onClick={() => setOpen((current) => !current)}
        aria-label="Ustawienia"
        aria-expanded={open}
        aria-controls={panelId}
        title="Ustawienia"
      >
        <SettingsIcon />
      </button>

      {open ? (
        <div className="settings-popover" id={panelId} role="dialog" aria-label="Ustawienia">
          <div className="settings-popover-header">
            <h2>Ustawienia</h2>
            <button
              type="button"
              className="settings-close"
              onClick={() => setOpen(false)}
              aria-label="Zamknij ustawienia"
            >
              ✕
            </button>
          </div>

          {hint ? <p className="voice-hint">{hint}</p> : null}

          <label className="mic-picker">
            <span>Mikrofon</span>
            <select
              value={deviceId}
              disabled={busy || !studentReady}
              onChange={(event) => onDeviceChange(event.target.value)}
            >
              {devices.length === 0 ? (
                <option value="">Kliknij Wilgusia (zgoda na mic)</option>
              ) : (
                devices.map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label}
                  </option>
                ))
              )}
            </select>
          </label>

          <button
            type="button"
            className="toolbar-button settings-action"
            onClick={() => {
              onNewConversation()
              setOpen(false)
            }}
            disabled={busy || !studentReady}
            title="Wyczyść historię rozmowy"
          >
            Nowa rozmowa
          </button>
        </div>
      ) : null}
    </div>
  )
}
