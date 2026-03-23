export type AppStatus =
  | 'IDLE'
  | 'LISTENING_WAKE_WORD'
  | 'RECORDING'
  | 'PROCESSING'

export type ChatRole = 'user' | 'wilga'

export interface ChatMessage {
  id: string
  timestamp: string
  role: ChatRole
  content: string
}
