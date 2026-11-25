/**
 * Decodes browser-recorded audio blobs (WebM/Ogg/MP4) using Web Audio API and
 * resamples to 16 kHz Mono 16-bit PCM WAV so live microphone recordings & watch
 * chunks run reliably without format conversion errors.
 */

export const RAW_MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: false,
  autoGainControl: true,
  channelCount: 1,
}

export function startContinuousPcmMonitor(
  stream: MediaStream,
  chunkMs: number,
  onChunk: (file: File) => void,
): () => void {
  const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
  const ctx = new AudioCtx()
  const source = ctx.createMediaStreamSource(stream)
  const processor = ctx.createScriptProcessor(4096, 1, 1)
  const silentGain = ctx.createGain()
  silentGain.gain.value = 0

  const nativeSr = ctx.sampleRate
  const warmupSamples = Math.round(nativeSr * 0.25) // skip initial hardware switch-on transient
  const targetChunkSamples = Math.max(1600, Math.round((nativeSr * chunkMs) / 1000))
  let skipped = 0
  let chunks: Float32Array[] = []
  let buffered = 0
  let stopped = false

  processor.onaudioprocess = (ev) => {
    if (stopped) return
    const input = ev.inputBuffer.getChannelData(0)
    if (skipped < warmupSamples) {
      const needSkip = Math.min(input.length, warmupSamples - skipped)
      skipped += needSkip
      if (needSkip >= input.length) return
      const rem = new Float32Array(input.subarray(needSkip))
      chunks.push(rem)
      buffered += rem.length
    } else {
      const copy = new Float32Array(input)
      chunks.push(copy)
      buffered += copy.length
    }

    if (buffered >= targetChunkSamples) {
      const merged = new Float32Array(buffered)
      let pos = 0
      for (const c of chunks) {
        merged.set(c, pos)
        pos += c.length
      }
      chunks = []
      buffered = 0
      const resampled = resampleLinear(merged, nativeSr, 16000)
      const wavBytes = encodePcm16Wav(resampled, 16000)
      onChunk(new File([wavBytes], `watch-${Date.now()}.wav`, { type: 'audio/wav' }))
    }
  }

  source.connect(processor)
  processor.connect(silentGain)
  silentGain.connect(ctx.destination)

  return () => {
    stopped = true
    try { processor.disconnect() } catch {}
    try { source.disconnect() } catch {}
    try { silentGain.disconnect() } catch {}
    ctx.close().catch(() => {})
  }
}

function resampleLinear(samples: Float32Array, origSr: number, targetSr: number): Float32Array {
  if (origSr === targetSr || samples.length < 2) return samples
  const outLen = Math.max(1, Math.round((samples.length * targetSr) / origSr))
  const out = new Float32Array(outLen)
  const ratio = (samples.length - 1) / Math.max(1, outLen - 1)
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio
    const lo = Math.floor(srcIdx)
    const hi = Math.min(samples.length - 1, lo + 1)
    const frac = srcIdx - lo
    out[i] = samples[lo] * (1 - frac) + samples[hi] * frac
  }
  return out
}

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
    // Soft boost faint microphone signals so VAD and speech recognition receive optimal level
    let maxAbs = 0
    for (let i = 0; i < mono.length; i++) {
      const a = Math.abs(mono[i])
      if (a > maxAbs) maxAbs = a
    }
    if (maxAbs > 0.001 && maxAbs < 0.45) {
      const boost = Math.min(5.0, 0.72 / maxAbs)
      for (let i = 0; i < mono.length; i++) {
        mono[i] = mono[i] * boost
      }
    }
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
