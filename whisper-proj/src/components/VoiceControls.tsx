import type { AppStatus } from '../types'

interface VoiceControlsProps {
  status: AppStatus
  isRecording: boolean
  isWakeWordSupported: boolean
  isWakeWordListening: boolean
  onToggleRecording: () => void
}

const statusClassName: Record<AppStatus, string> = {
  IDLE: 'status-idle',
  LISTENING_WAKE_WORD: 'status-listening',
  RECORDING: 'status-recording',
  PROCESSING: 'status-processing',
}

export function VoiceControls({
  status,
  isRecording,
  isWakeWordSupported,
  isWakeWordListening,
  onToggleRecording,
}: VoiceControlsProps) {
  const buttonLabel = isRecording ? 'Zatrzymaj' : 'Rozpocznij nasłuchiwanie'
  const isButtonDisabled = status === 'PROCESSING'

  let statusText = ''
  if (status === 'LISTENING_WAKE_WORD') {
    statusText = isWakeWordListening
      ? "Oczekuję na 'hej wilga'..."
      : 'Inicjalizuję nasłuchiwanie...'
  }
  if (status === 'RECORDING') {
    statusText = 'Nagrywam...'
  }
  if (status === 'PROCESSING') {
    statusText = 'Przetwarzam i generuję odpowiedź...'
  }
  if (status === 'IDLE') {
    statusText = isWakeWordSupported
      ? 'Tryb gotowości. Możesz użyć przycisku.'
      : 'Tryb ręczny. Web Speech API nie jest dostępne.'
  }

  return (
    <section className="voice-controls">
      <button
        type="button"
        className={`record-button ${isRecording ? 'record-button-stop' : 'record-button-start'}`}
        onClick={onToggleRecording}
        disabled={isButtonDisabled}
      >
        {buttonLabel}
      </button>
      <p className={`status-chip ${statusClassName[status]}`}>{statusText}</p>
    </section>
  )
}
