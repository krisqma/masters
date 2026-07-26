const WAKE_PHRASE_PATTERNS = [
  'hej wilgus',
  'hej wilga',
  'hej wilgo',
  'hej wilko',
  'ej wilgus',
  'ej wilga',
  'ej wilgo',
  'ej wilko',
  'hey wilgus',
  'hey wilga',
]

const normalize = (value: string) =>
  value
    .toLocaleLowerCase('pl-PL')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

/** Remove a leading wake phrase so Whisper+chat don't treat it as the question. */
export function stripLeadingWakePhrase(transcript: string): string {
  const original = transcript.trim()
  if (!original) {
    return original
  }

  const normalized = normalize(original)
  for (const phrase of WAKE_PHRASE_PATTERNS) {
    if (normalized === phrase) {
      return ''
    }
    if (normalized.startsWith(`${phrase} `)) {
      // Map back to original by cutting roughly the same word count from the start.
      const phraseWordCount = phrase.split(' ').length
      const words = original.split(/\s+/)
      return words.slice(phraseWordCount).join(' ').trim()
    }
  }

  const fuzzy = normalized.match(/^(?:h|e)j\s+wil(g|k)\w*\s+(.*)$/)
  if (fuzzy?.[2]) {
    const words = original.split(/\s+/)
    return words.slice(2).join(' ').trim()
  }

  return original
}
