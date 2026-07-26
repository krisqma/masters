import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../types'

interface ConsoleWindowProps {
  messages: ChatMessage[]
}

const roleClassName: Record<ChatMessage['role'], string> = {
  system: 'console-line-system',
  user: 'console-line-user',
  wilga: 'console-line-wilga',
}

const roleLabel: Record<ChatMessage['role'], string> = {
  system: 'System',
  user: 'Ty',
  wilga: 'Wilga',
}

export function ConsoleWindow({ messages }: ConsoleWindowProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    container.scrollTo({
      top: container.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages])

  return (
    <section className="console-window" aria-live="polite" aria-label="Konsola transkrypcji">
      <header className="console-header">conversation.log</header>
      <div ref={containerRef} className="console-content">
        {messages.length === 0 ? (
          <p className="console-empty">
            Oczekiwanie na pierwszą wiadomość. Powiedz „hej wilguś” albo kliknij Wilgusia.
          </p>
        ) : (
          <ul className="console-list">
            {messages.map((record) => (
              <li key={record.id} className={`console-line ${roleClassName[record.role]}`}>
                <span className="console-line-time">
                  [{record.timestamp}] {roleLabel[record.role]}:
                </span>
                <span className="console-line-text">{record.content}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
