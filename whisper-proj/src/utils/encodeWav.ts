/** Decode arbitrary browser audio (webm/mp4) and re-encode as PCM WAV for Whisper/ffmpeg. */

const writeString = (view: DataView, offset: number, value: string) => {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}

const encodeWavFromAudioBuffer = (buffer: AudioBuffer): Blob => {
  const sampleRate = buffer.sampleRate
  const channelCount = 1
  const samples = buffer.getChannelData(0)
  // Mixdown if multi-channel
  let mono = samples
  if (buffer.numberOfChannels > 1) {
    const mixed = new Float32Array(buffer.length)
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const data = buffer.getChannelData(channel)
      for (let index = 0; index < data.length; index += 1) {
        mixed[index] += data[index] / buffer.numberOfChannels
      }
    }
    mono = mixed
  }

  const bytesPerSample = 2
  const blockAlign = channelCount * bytesPerSample
  const dataSize = mono.length * bytesPerSample
  const headerSize = 44
  const arrayBuffer = new ArrayBuffer(headerSize + dataSize)
  const view = new DataView(arrayBuffer)

  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(view, 8, 'WAVE')
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, channelCount, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bytesPerSample * 8, true)
  writeString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  let offset = 44
  for (let index = 0; index < mono.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, mono[index] ?? 0))
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
    offset += 2
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' })
}

export async function blobToWav(blob: Blob): Promise<Blob> {
  const audioContextCtor =
    window.AudioContext ??
    (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!audioContextCtor) {
    throw new Error('Brak AudioContext — nie mogę skonwertować nagrania do WAV.')
  }

  const context = new audioContextCtor()
  try {
    const raw = await blob.arrayBuffer()
    const decoded = await context.decodeAudioData(raw.slice(0))
    return encodeWavFromAudioBuffer(decoded)
  } finally {
    await context.close().catch(() => undefined)
  }
}
