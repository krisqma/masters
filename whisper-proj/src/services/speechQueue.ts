import { playAudioBlob, synthesizeSpeech } from './ttsService'

export interface SpeechQueueOptions {
  onSpeakingStart?: () => void
  onIdle?: () => void
  onError?: (message: string) => void
}

const MIN_SENTENCE_CHARS = 8

/** Pull complete sentences from a streaming buffer; leave the unfinished tail. */
export function extractSentences(buffer: string): { sentences: string[]; rest: string } {
  const normalized = buffer.replace(/\s+/g, ' ').trimStart()
  const sentences: string[] = []
  let cursor = 0
  const boundary = /[.!?…]+\s+/g
  let match: RegExpExecArray | null

  while ((match = boundary.exec(normalized)) !== null) {
    const end = match.index + match[0].length
    const sentence = normalized.slice(cursor, end).trim()
    if (sentence.length >= MIN_SENTENCE_CHARS) {
      sentences.push(sentence)
      cursor = end
    }
  }

  return {
    sentences,
    rest: normalized.slice(cursor),
  }
}

export class SpeechQueue {
  private readonly queue: string[] = []
  private running = false
  private aborted = false
  private abortController: AbortController | null = null
  private speakingStarted = false
  private readonly options: SpeechQueueOptions

  constructor(options: SpeechQueueOptions = {}) {
    this.options = options
  }

  enqueue(text: string): void {
    const trimmed = text.trim()
    if (!trimmed || this.aborted) {
      return
    }
    this.queue.push(trimmed)
    void this.pump()
  }

  cancel(): void {
    this.aborted = true
    this.queue.length = 0
    this.abortController?.abort()
    this.abortController = null
    this.running = false
    this.speakingStarted = false
  }

  reset(): void {
    this.cancel()
    this.aborted = false
  }

  get isBusy(): boolean {
    return this.running || this.queue.length > 0
  }

  private async pump(): Promise<void> {
    if (this.running) {
      return
    }
    this.running = true

    while (this.queue.length > 0 && !this.aborted) {
      const next = this.queue.shift()
      if (!next) {
        continue
      }

      this.abortController = new AbortController()
      try {
        if (!this.speakingStarted) {
          this.speakingStarted = true
          this.options.onSpeakingStart?.()
        }
        const blob = await synthesizeSpeech(next, this.abortController.signal)
        if (this.aborted) {
          break
        }
        await playAudioBlob(blob, this.abortController.signal)
      } catch (error) {
        if (this.aborted || (error instanceof DOMException && error.name === 'AbortError')) {
          break
        }
        const message = error instanceof Error ? error.message : 'Błąd syntezy mowy.'
        this.options.onError?.(message)
      } finally {
        this.abortController = null
      }
    }

    this.running = false
    if (!this.aborted && this.queue.length === 0) {
      this.speakingStarted = false
      this.options.onIdle?.()
    }
  }
}
