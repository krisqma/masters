import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../types'

interface ChatTranscriptProps {
  messages: ChatMessage[]
}

export function ChatTranscript({ messages }: ChatTranscriptProps) {
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
    <section className="chat-transcript" aria-live="polite" aria-label="Rozmowa">
      <header className="chat-transcript-header">
        <h2>Rozmowa</h2>
      </header>
      <div ref={containerRef} className="chat-transcript-scroll">
        {messages.length === 0 ? (
          <p className="chat-empty">
            Powiedz „hej wilguś” albo kliknij Wilgusia, żeby zacząć.
          </p>
        ) : (
          <ul className="chat-list">
            {messages.map((record) => {
              if (record.role === 'system') {
                return (
                  <li key={record.id} className="chat-note">
                    <span>{record.content}</span>
                  </li>
                )
              }

              const isUser = record.role === 'user'
              return (
                <li
                  key={record.id}
                  className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-wilga'}`}
                >
                  <span className="chat-bubble-meta">
                    {isUser ? 'Ty' : 'Wilguś'} · {record.timestamp}
                  </span>
                  <p>{record.content || '…'}</p>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </section>
  )
}
