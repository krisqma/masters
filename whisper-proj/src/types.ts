export type AppStatus =
  | 'IDLE'
  | 'LISTENING_WAKE_WORD'
  | 'LISTENING_FOLLOW_UP'
  | 'RECORDING'
  | 'PROCESSING'
  | 'SPEAKING'

export type ChatRole = 'user' | 'wilga' | 'system'

export interface ChatMessage {
  id: string
  timestamp: string
  role: ChatRole
  content: string
}
