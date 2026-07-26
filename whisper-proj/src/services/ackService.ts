import { ACK_PHRASES, type AckPhrase } from '../data/ackPhrases'

let currentAudio: HTMLAudioElement | null = null
let lastId: string | null = null

const pickPhrase = (): AckPhrase => {
  if (ACK_PHRASES.length === 0) {
    throw new Error('Brak fraz ack.')
  }
  if (ACK_PHRASES.length === 1) {
    return ACK_PHRASES[0]!
  }
  let phrase = ACK_PHRASES[Math.floor(Math.random() * ACK_PHRASES.length)]!
  // Avoid immediate repeat of the same clip.
  for (let attempt = 0; attempt < 4 && phrase.id === lastId; attempt += 1) {
    phrase = ACK_PHRASES[Math.floor(Math.random() * ACK_PHRASES.length)]!
  }
  lastId = phrase.id
  return phrase
}

export function stopAck(): void {
  if (!currentAudio) {
    return
  }
  currentAudio.pause()
  currentAudio.currentTime = 0
  currentAudio = null
}

/** Play a random local ack MP3. Returns the phrase text for UI status. */
export function playAck(): string {
  stopAck()
  const phrase = pickPhrase()
  const audio = new Audio(phrase.src)
  audio.preload = 'auto'
  currentAudio = audio
  void audio.play().catch(() => {
    if (currentAudio === audio) {
      currentAudio = null
    }
  })
  audio.addEventListener(
    'ended',
    () => {
      if (currentAudio === audio) {
        currentAudio = null
      }
    },
    { once: true },
  )
  return phrase.text
}
