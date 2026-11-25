/**
 * Decodes browser-recorded audio blobs (WebM/Ogg/MP4) using Web Audio API and
 * resamples to 16 kHz Mono 16-bit PCM WAV so live microphone recordings & watch
 * chunks run reliably without format conversion errors.
 */
export async function blobToWavFile(blob: Blob, baseName = 'mic-recording'): Promise<File> {
  const arrayBuf = await blob.arrayBuffer()
  const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
  const ctx = new AudioCtx()
  try {
    const decoded = await ctx.decodeAudioData(arrayBuf)
    const targetSr = 16000
    const duration = decoded.duration || (decoded.length / decoded.sampleRate)
    const targetLen = Math.max(1, Math.round(duration * targetSr))
    const offline = new OfflineAudioContext(1, targetLen, targetSr)
    const src = offline.createBufferSource()
    src.buffer = decoded
    src.connect(offline.destination)
    src.start(0)
    const rendered = await offline.startRendering()
    const mono = rendered.getChannelData(0)
    const wavBytes = encodePcm16Wav(mono, targetSr)
    return new File([wavBytes], `${baseName}-${Date.now()}.wav`, { type: 'audio/wav' })
  } finally {
    await ctx.close().catch(() => {})
  }
}

function encodePcm16Wav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const dataBytes = samples.length * 2
  const buf = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buf)

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i))
  }

  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, 1, true) // Mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, 'data')
  view.setUint32(40, dataBytes, true)

  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
    offset += 2
  }
  return buf
}
