import { WILGUS_BY_STATUS } from '../data/wilgusStates'
import type { AppStatus } from '../types'

interface WilgusAssistantProps {
  status: AppStatus
  bubbleText: string | null
  disabled?: boolean
  studentReady?: boolean
  onClick: () => void
  inputLevel?: number
}

export function WilgusAssistant({
  status,
  bubbleText,
  disabled = false,
  studentReady = true,
  onClick,
  inputLevel = 0,
}: WilgusAssistantProps) {
  const visual = studentReady
    ? WILGUS_BY_STATUS[status]
    : WILGUS_BY_STATUS.PROCESSING
  const label = studentReady ? visual.label : 'Rozgrzewam model…'
  const levelPercent = Math.round(Math.min(1, Math.max(0, inputLevel)) * 100)
  const showLevel = studentReady && status === 'RECORDING'
  const isDisabled = disabled || !studentReady

  return (
    <div className="wilgus-stage">
      {bubbleText ? (
        <div className="wilgus-bubble" role="status">
          <p>{bubbleText}</p>
        </div>
      ) : null}

      <button
        type="button"
        className={`wilgus-button ${visual.ringClass} ${isDisabled ? 'wilgus-button-disabled' : ''}`}
        onClick={onClick}
        disabled={isDisabled}
        aria-label={label}
        title={label}
      >
        <img src={visual.src} alt="Wilguś" className="wilgus-image" draggable={false} />
        {showLevel ? (
          <span className="wilgus-level" aria-hidden>
            {levelPercent}%
          </span>
        ) : null}
      </button>

      <p className="wilgus-caption">{label}</p>
    </div>
  )
}
