const STORAGE_KEY = 'wilga.audioInputDeviceId'

export interface AudioInputDevice {
  deviceId: string
  label: string
}

const BAD_MIC_RE =
  /blackhole|virtual|continuity|iphone|ipad|iphone|zoom|teams|skype|aggregate|soundflower|loopback|cable/i
const GOOD_MIC_RE = /macbook|built-?in|wbudowan|internal|mikrofon.*mac|default/i

export function scoreAudioInputLabel(label: string): number {
  let score = 0
  if (GOOD_MIC_RE.test(label)) score += 100
  if (BAD_MIC_RE.test(label)) score -= 100
  if (!label || /^microphone\s*\d*$/i.test(label)) score -= 5
  return score
}

export async function ensureMicPermission(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  stream.getTracks().forEach((track) => track.stop())
}

export async function listAudioInputDevices(): Promise<AudioInputDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return []
  }

  // Labels are empty until permission was granted at least once.
  try {
    await ensureMicPermission()
  } catch {
    // Still try enumerate — may return anonymous devices.
  }

  const devices = await navigator.mediaDevices.enumerateDevices()
  return devices
    .filter((device) => device.kind === 'audioinput')
    .map((device, index) => ({
      deviceId: device.deviceId,
      label: device.label?.trim() || `Mikrofon ${index + 1}`,
    }))
}

export function pickPreferredDeviceId(devices: AudioInputDevice[]): string {
  if (devices.length === 0) {
    return ''
  }

  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored && devices.some((device) => device.deviceId === stored)) {
    return stored
  }

  const ranked = [...devices].sort(
    (a, b) => scoreAudioInputLabel(b.label) - scoreAudioInputLabel(a.label),
  )
  return ranked[0]?.deviceId ?? ''
}

export function persistAudioInputDeviceId(deviceId: string): void {
  if (!deviceId) {
    localStorage.removeItem(STORAGE_KEY)
    return
  }
  localStorage.setItem(STORAGE_KEY, deviceId)
}

export function buildAudioConstraints(deviceId?: string): MediaTrackConstraints {
  const base: MediaTrackConstraints = {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  }
  if (deviceId) {
    return { ...base, deviceId: { exact: deviceId } }
  }
  return base
}
