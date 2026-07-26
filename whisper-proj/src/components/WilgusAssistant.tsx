import { WILGUS_BY_STATUS } from '../data/wilgusStates'
import type { AppStatus } from '../types'

interface WilgusAssistantProps {
  status: AppStatus
  bubbleText: string | null
  disabled?: boolean
  onClick: () => void
  inputLevel?: number
}

export function WilgusAssistant({
  status,
  bubbleText,
  disabled = false,
  onClick,
  inputLevel = 0,
}: WilgusAssistantProps) {
  const visual = WILGUS_BY_STATUS[status]
  const levelPercent = Math.round(Math.min(1, Math.max(0, inputLevel)) * 100)
  const showLevel = status === 'RECORDING'

  return (
    <div className="wilgus-stage">
      {bubbleText ? (
        <div className="wilgus-bubble" role="status">
          <p>{bubbleText}</p>
        </div>
      ) : null}

      <button
        type="button"
        className={`wilgus-button ${visual.ringClass} ${disabled ? 'wilgus-button-disabled' : ''}`}
        onClick={onClick}
        disabled={disabled}
        aria-label={visual.label}
        title={visual.label}
      >
        <img src={visual.src} alt="Wilguś" className="wilgus-image" draggable={false} />
        {showLevel ? (
          <span className="wilgus-level" aria-hidden>
            {levelPercent}%
          </span>
        ) : null}
      </button>

      <p className="wilgus-caption">{visual.label}</p>
    </div>
  )
}
