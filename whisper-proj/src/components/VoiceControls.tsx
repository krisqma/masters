import type { AudioInputDevice } from '../utils/audioDevices'
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
}: VoiceControlsProps) {
  const busy =
    status === 'RECORDING' || status === 'PROCESSING' || status === 'SPEAKING'

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

  return (
    <section className="voice-toolbar">
      <div className="voice-toolbar-left">
        {hint ? <p className="voice-hint">{hint}</p> : null}
        <label className="mic-picker">
          <span>Mikrofon</span>
          <select
            value={deviceId}
            disabled={busy}
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
      </div>

      <button
        type="button"
        className="toolbar-button"
        onClick={onNewConversation}
        disabled={busy}
        title="Wyczyść historię rozmowy"
      >
        Nowa rozmowa
      </button>
    </section>
  )
}
