import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileAudio,
  Mic,
  Send,
  ShieldAlert,
  ShieldCheck,
  Square,
  Upload,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import {
  authenticateVoice,
  sendAgentTurn,
  sendWatchChunk,
} from '../api'
import type { AlertItem } from '../types'
import { RAW_MIC_CONSTRAINTS, blobToWavFile, startContinuousPcmMonitor } from '../utils/wavEncoder'
import { WavePlayer } from './WavePlayer'

type TurnItem = {
  role: 'user' | 'agent'
  text: string
  sounds?: any[]
  pitch?: number
  matchedSessions?: { session_id: number; filename: string; summary: string }[]
}

export function AgentView() {
  const [turns, setTurns] = useState<TurnItem[]>([
    {
      role: 'agent',
      text: 'SonicVoice Copilot active. Ask questions about vocal emphasis, speaker delivery, or acoustic room events across your recordings.',
    },
  ])
  const [text, setText] = useState('')
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const [speakReply, setSpeakReply] = useState(true)
  const [denoise, setDenoise] = useState(true)
  const [err, setErr] = useState('')
  const recRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  function speakText(msg: string) {
    if (!speakReply || !('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(msg)
    u.rate = 1.03
    window.speechSynthesis.speak(u)
  }

  async function submitTurn(file?: File, userInput?: string) {
    setBusy(true)
    setErr('')
    try {
      const history = turns.slice(-6).map((t) => ({ role: t.role, text: t.text }))
      const res = await sendAgentTurn({ file, text: userInput, denoise, history })
      setTurns((prev) => [
        ...prev,
        {
          role: 'user',
          text: res.transcript || userInput || '(ambient audio turn)',
          sounds: res.sound_events,
          pitch: res.pitch_stats?.mean_f0_hz,
        },
        {
          role: 'agent',
          text: res.reply,
          sounds: res.new_notes || res.new_reminders,
          matchedSessions: res.matched_sessions,
        },
      ])
      speakText(res.reply)
    } catch (e: any) {
      setErr(e.message || 'Voice turn failed')
    } finally {
      setBusy(false)
    }
  }

  async function toggleMic() {
    if (recording) {
      recRef.current?.stop()
      setRecording(false)
      return
    }
    setErr('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const rec = new MediaRecorder(stream)
      recRef.current = rec
      chunksRef.current = []
      rec.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data) }
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || 'audio/webm' })
        if (!blob.size) return
        const wav = await blobToWavFile(blob, 'voice-turn')
        await submitTurn(wav)
      }
      rec.start()
      setRecording(true)
    } catch {
      setErr('Microphone permission denied or unavailable.')
    }
  }

  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), [])

  return (
    <div className="content">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>Voice Copilot — Conversational Audio Intelligence</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>Ask questions about acoustic emphasis, tone of voice, speaker turns, or sound context.</p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <label className="toggle-chip">
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
              DSP Denoise
            </label>
            <button className="secondary-btn" onClick={() => setSpeakReply((v) => !v)}>
              {speakReply ? <Volume2 size={14} /> : <VolumeX size={14} />}
              {speakReply ? 'TTS Speech On' : 'TTS Muted'}
            </button>
          </div>
        </div>

        <div className="tag-row" style={{ marginBottom: 12 }}>
          {[
            'Summarize key spoken takeaways',
            'Did anyone ask questions or sound stressed?',
            'Spot high-priority vocal emphasis moments',
          ].map((preset) => (
            <button key={preset} className="secondary-btn" style={{ fontSize: 11.5, padding: '5px 12px' }} disabled={busy} onClick={() => submitTurn(undefined, preset)}>
              {preset}
            </button>
          ))}
        </div>

        {err && <div className="notice-bar">{err}</div>}

        <div className="chat-box">
          {turns.map((t, i) => (
            <div key={i} className={`chat-msg ${t.role}`}>
              <div className="mono" style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
                {t.role === 'user' ? 'SPEAKER' : 'SONIC COPILOT'}
                {t.pitch ? ` · Mean F0: ${t.pitch} Hz` : ''}
              </div>
              <div>{t.text}</div>
              {t.role === 'user' && t.sounds && t.sounds.length > 0 && (
                <div className="tag-row">
                  {t.sounds.map((s, idx) => (
                    <span key={idx} className="sound-chip">[{s.label}] {Math.round((s.confidence || 0) * 100)}%</span>
                  ))}
                </div>
              )}
              {t.role === 'agent' && t.matchedSessions && t.matchedSessions.length > 0 && (
                <div className="tag-row">
                  {t.matchedSessions.map((m) => (
                    <span key={m.session_id} className="tag">Source: {m.filename}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="input-bar">
          <button className={recording ? 'danger-btn' : 'primary-btn'} onClick={toggleMic} disabled={busy}>
            {recording ? <Square size={14} /> : <Mic size={14} />}
            {recording ? 'Stop & Send' : 'Speak to Copilot'}
          </button>
          <input
            className="input-field"
            value={text}
            placeholder="Type or speak a question about your audio..."
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && text.trim() && !busy) {
                const msg = text.trim()
                setText('')
                submitTurn(undefined, msg)
              }
            }}
          />
          <button
            className="secondary-btn"
            disabled={busy || !text.trim()}
            onClick={() => {
              const msg = text.trim()
              setText('')
              submitTurn(undefined, msg)
            }}
          >
            <Send size={14} /> Send
          </button>
        </div>
      </div>
    </div>
  )
}

export function WatchView({ onAlertLogged }: { onAlertLogged: () => void }) {
  const [watching, setWatching] = useState(false)
  const [denoise, setDenoise] = useState(true)
  const [baselineRms, setBaselineRms] = useState(0)
  const [calibrating, setCalibrating] = useState(false)
  const [rmsDb, setRmsDb] = useState(-60)
  const [speechActive, setSpeechActive] = useState(false)
  const [urgent, setUrgent] = useState<AlertItem | null>(null)
  const [feed, setFeed] = useState<{ ts: string; msg: string; rms: number }[]>([])
  const [err, setErr] = useState('')
  const stopCaptureRef = useRef<(() => void) | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const denoiseRef = useRef(denoise)
  const baselineRef = useRef(baselineRms)
  const calibratingRef = useRef(calibrating)
  const inFlightRef = useRef(false)
  useEffect(() => { denoiseRef.current = denoise }, [denoise])
  useEffect(() => { baselineRef.current = baselineRms }, [baselineRms])
  useEffect(() => { calibratingRef.current = calibrating }, [calibrating])

  async function handleChunk(wav: File) {
    if (inFlightRef.current) return
    inFlightRef.current = true
    try {
      const res = await sendWatchChunk(wav, denoiseRef.current, baselineRef.current)
      setRmsDb(res.rms_db)
      if (calibratingRef.current && res.noise_rms != null) {
        setBaselineRms(res.noise_rms)
        setCalibrating(false)
        setSpeechActive(false)
        setFeed((prev) => [{ ts: new Date().toLocaleTimeString(), msg: `Calibrated room floor (${res.rms_db.toFixed(1)} dB)`, rms: res.rms_db }, ...prev].slice(0, 15))
        return
      }
      setSpeechActive(Boolean(res.has_speech))
      if (res.urgent_alert) {
        setUrgent(res.urgent_alert)
        onAlertLogged()
      }
      const tagText = res.events?.length
        ? res.events.map((e: any) => `${e.label} (${Math.round(e.confidence * 100)}%)`).join(', ')
        : res.has_speech ? 'Speech active (Silero VAD)' : 'Quiet / stationary ambient floor'
      setFeed((prev) => [{ ts: new Date().toLocaleTimeString(), msg: tagText, rms: res.rms_db }, ...prev].slice(0, 15))
    } catch (e: any) {
      setErr(e.message || 'Watch chunk failed')
    } finally {
      inFlightRef.current = false
    }
  }

  async function toggleWatch() {
    if (watching) {
      stopCaptureRef.current?.()
      stopCaptureRef.current = null
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setWatching(false)
      setCalibrating(false)
      setSpeechActive(false)
      return
    }
    setErr('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW_MIC_CONSTRAINTS })
      streamRef.current = stream
      setWatching(true)
      stopCaptureRef.current = startContinuousPcmMonitor(stream, 2500, handleChunk)
    } catch {
      setErr('Microphone permission required for Live Monitor.')
    }
  }

  useEffect(() => () => {
    stopCaptureRef.current?.()
    streamRef.current?.getTracks().forEach((t) => t.stop())
  }, [])

  const pct = Math.max(4, Math.min(100, ((rmsDb + 65) / 65) * 100))
  const floorDb = baselineRms > 0 ? `${(20 * Math.log10(baselineRms)).toFixed(1)} dB` : 'Auto (-33 dB)'

  return (
    <div className="content">
      {urgent && (
        <div className="urgent-bar">
          <ShieldAlert size={18} />
          <span>ALERT: {urgent.event_label} ({Math.round(urgent.confidence * 100)}%) — {urgent.snippet_text}</span>
          <button className="secondary-btn" style={{ marginLeft: 'auto' }} onClick={() => setUrgent(null)}>Acknowledge</button>
        </div>
      )}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>Live Room Acoustic HUD</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>Real-time streaming PCM chunks through smoothed Spectral Subtraction, calibrated ambient noise gate, Silero VAD, and Sound CNN.</p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <label className="toggle-chip">
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
              Denoise Stream
            </label>
            {watching && (
              <button className="secondary-btn" onClick={() => setCalibrating(true)} disabled={calibrating}>
                {calibrating ? 'Sampling Room...' : 'Calibrate Fan Floor'}
              </button>
            )}
            <button className={watching ? 'danger-btn' : 'primary-btn'} onClick={toggleWatch}>
              {watching ? <Square size={14} /> : <Mic size={14} />}
              {watching ? 'Stop Monitor' : 'Start Monitor'}
            </button>
          </div>
        </div>
        {err && <div className="notice-bar">{err}</div>}
        <div>
          <div className="row-between" style={{ fontSize: 12 }}>
            <span>Level: <b className="mono" style={{ color: '#06b6d4' }}>{rmsDb.toFixed(1)} dB</b> · Noise Floor: <b className="mono">{floorDb}</b></span>
            <span>VAD: <b style={{ color: speechActive ? '#10b981' : '#64748b' }}>{speechActive ? 'Speech Active' : 'Idle'}</b></span>
          </div>
          <div className="meter-bar"><div className="meter-fill" style={{ width: `${pct}%` }} /></div>
        </div>
        <h4 style={{ margin: '18px 0 10px', fontSize: 13 }}>Live Event Stream</h4>
        {feed.length === 0 ? (
          <p className="muted">Start the monitor and click &ldquo;Calibrate Fan Floor&rdquo; in a noisy classroom, then test speech, clapping, knocking, or laughter.</p>
        ) : (
          feed.map((f, i) => (
            <div key={i} className="list-card">
              <div><b>{f.msg}</b><div className="muted mono">{f.ts}</div></div>
              <span className="tag mono">{f.rms.toFixed(1)} dB</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export function VoiceAuthenticatorView() {
  const [file, setFile] = useState<File | null>(null)
  const [audioUrl, setAudioUrl] = useState('')
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [busy, setBusy] = useState(false)
  const [denoise, setDenoise] = useState(true)
  const [result, setResult] = useState<any>(null)
  const [err, setErr] = useState('')
  const [drag, setDrag] = useState(false)

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    if (timer.current) clearInterval(timer.current)
  }, [audioUrl])

  function chooseFile(f?: File) {
    if (!f) return
    setErr('')
    setFile(f)
    setResult(null)
    setAudioUrl(URL.createObjectURL(f))
  }

  async function startRecording() {
    setErr('')
    setResult(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW_MIC_CONSTRAINTS })
      streamRef.current = stream
      const rec = new MediaRecorder(stream)
      mediaRecorder.current = rec
      chunksRef.current = []
      rec.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data) }
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || 'audio/webm' })
        if (!blob.size) return
        const wav = await blobToWavFile(blob, 'voice-sample')
        setFile(wav)
        setAudioUrl(URL.createObjectURL(wav))
      }
      rec.start()
      setRecording(true)
      setSeconds(0)
      timer.current = window.setInterval(() => setSeconds((s) => s + 1), 1000)
    } catch {
      setErr('Microphone permission required for voice authentication.')
    }
  }

  function stopRecording() {
    if (timer.current) clearInterval(timer.current)
    if (mediaRecorder.current?.state === 'recording') mediaRecorder.current.stop()
    setRecording(false)
  }

  async function runScan() {
    if (!file) return
    setBusy(true)
    setErr('')
    try {
      const res = await authenticateVoice(file, denoise)
      setResult(res)
    } catch (e: any) {
      setErr(e.message || 'Authenticity analysis failed')
    } finally {
      setBusy(false)
    }
  }

  const fmt = (s: number) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${Math.floor(s % 60).toString().padStart(2, '0')}`

  return (
    <div className="content">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>AI Voice Authenticator — Deepfake &amp; Synthetic Speech Detection</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>
              Forensic acoustic inspection of micro-pitch jitter, vocal shimmer, and LPC glottal excitation to distinguish natural human vocal tracts from AI neural vocoders.
            </p>
          </div>
          <label className="toggle-chip">
            <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
            Denoise Signal
          </label>
        </div>

        {err && <div className="notice-bar">{err}</div>}

        <div className="grid-2" style={{ marginBottom: 16 }}>
          <div
            className={`dropzone ${drag ? 'dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); chooseFile(e.dataTransfer.files[0]) }}
          >
            <div className="mic-badge"><Upload size={20} /></div>
            <h3>Upload Voice Sample</h3>
            <p className="muted" style={{ margin: '4px 0 12px' }}>Upload a suspect voice clip or AI audio sample (WAV, MP3, M4A, WebM).</p>
            <label className="secondary-btn">
              Choose File
              <input type="file" accept="audio/*" hidden onChange={(e) => chooseFile(e.target.files?.[0])} />
            </label>
          </div>

          <div className="record-box">
            <div className={`mic-badge ${recording ? 'live' : ''}`}><Mic size={22} /></div>
            <h3>{recording ? 'Recording Voice...' : 'Record Microphone Sample'}</h3>
            <p className="muted" style={{ margin: '4px 0 0' }}>Speak naturally for 2–5 seconds to test your vocal tract acoustics.</p>
            <div className="record-clock">{fmt(seconds)}</div>
            {!recording ? (
              <button className="primary-btn" onClick={startRecording}><Mic size={14} /> Start Voice Capture</button>
            ) : (
              <button className="danger-btn" onClick={stopRecording}><Square size={14} /> Stop Capture</button>
            )}
          </div>
        </div>

        {file && (
          <div className="list-card" style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <FileAudio size={18} color="#06b6d4" />
              <div><b>{file.name}</b> <span className="muted mono">({(file.size / 1024).toFixed(1)} KB)</span></div>
            </div>
            <button className="primary-btn" onClick={runScan} disabled={busy}>
              {busy ? 'Scanning Biometrics...' : 'Run Authenticity Scan'}
            </button>
          </div>
        )}

        {audioUrl && <div style={{ marginBottom: 16 }}><WavePlayer audioUrl={audioUrl} /></div>}

        {result && (
          <div>
            <div
              className="panel"
              style={{
                background: result.is_synthetic
                  ? 'rgba(239, 68, 68, 0.08)'
                  : result.authenticity_score >= 68
                  ? 'rgba(16, 185, 129, 0.08)'
                  : 'rgba(245, 158, 11, 0.08)',
                borderColor: result.is_synthetic
                  ? 'rgba(239, 68, 68, 0.4)'
                  : result.authenticity_score >= 68
                  ? 'rgba(16, 185, 129, 0.4)'
                  : 'rgba(245, 158, 11, 0.4)',
              }}
            >
              <div className="row-between">
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  {result.is_synthetic ? (
                    <ShieldAlert size={28} color="#ef4444" />
                  ) : result.authenticity_score >= 68 ? (
                    <ShieldCheck size={28} color="#10b981" />
                  ) : (
                    <AlertTriangle size={28} color="#f59e0b" />
                  )}
                  <div>
                    <h2 style={{ margin: 0, fontSize: 18 }}>{result.verdict}</h2>
                    <span className="muted mono">Confidence: {Math.round((result.confidence || 0.8) * 100)}% · Risk Level: {result.risk_level?.toUpperCase()}</span>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <small className="muted" style={{ display: 'block', textTransform: 'uppercase', fontSize: 10 }}>Authenticity Index</small>
                  <b className="mono" style={{ fontSize: 24, color: result.authenticity_score >= 68 ? '#10b981' : result.is_synthetic ? '#ef4444' : '#f59e0b' }}>
                    {result.authenticity_score}%
                  </b>
                </div>
              </div>

              <div className="meter-bar" style={{ marginTop: 12 }}>
                <div
                  className="meter-fill"
                  style={{
                    width: `${result.authenticity_score}%`,
                    background: result.is_synthetic
                      ? '#ef4444'
                      : result.authenticity_score >= 68
                      ? '#10b981'
                      : '#f59e0b',
                  }}
                />
              </div>

              <p style={{ margin: '14px 0 0', fontSize: 13, lineHeight: 1.55 }}>{result.summary}</p>
            </div>

            {result.metrics && (
              <div className="metric-strip" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
                <div className="metric-cell">
                  <small>Pitch Jitter (F0 Perturbation)</small>
                  <b>{result.metrics.jitter_percent ?? '—'}%</b>
                  <span className="muted" style={{ fontSize: 11 }}>{result.metrics.jitter_status}</span>
                </div>
                <div className="metric-cell">
                  <small>Amplitude Shimmer (Energy)</small>
                  <b>{result.metrics.shimmer_db ?? '—'} dB</b>
                  <span className="muted" style={{ fontSize: 11 }}>{result.metrics.shimmer_status}</span>
                </div>
                <div className="metric-cell">
                  <small>LPC Glottal Kurtosis</small>
                  <b>{result.metrics.lpc_residual_kurtosis ?? '—'}</b>
                  <span className="muted" style={{ fontSize: 11 }}>{result.metrics.lpc_status}</span>
                </div>
                <div className="metric-cell">
                  <small>High-Band Spectral Roll-off</small>
                  <b>{result.metrics.hf_ratio ?? '—'}</b>
                  <span className="muted" style={{ fontSize: 11 }}>{result.metrics.hf_status}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export function EngineSpecsModal({ onClose }: { onClose: () => void }) {
  const rows = [
    ['clapping', '0.78', '0.88', '0.82', '8'],
    ['coughing', '0.86', '0.75', '0.80', '8'],
    ['dog', '0.62', '0.62', '0.62', '8'],
    ['door_wood_knock', '0.88', '0.88', '0.88', '8'],
    ['footsteps', '1.00', '0.75', '0.86', '8'],
    ['glass_breaking', '0.78', '0.88', '0.82', '8'],
    ['laughing', '0.57', '1.00', '0.73', '8'],
    ['rain', '0.89', '1.00', '0.94', '8'],
    ['siren', '1.00', '1.00', '1.00', '8'],
    ['sneezing', '1.00', '0.25', '0.40', '8'],
  ]
  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)', display: 'grid', placeItems: 'center', zIndex: 50, padding: 20 }}
      onClick={onClose}
    >
      <div className="panel" style={{ maxWidth: 640, width: '100%', maxHeight: '85vh', overflowY: 'auto' }} onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <div>
            <h3>SonicVoice Engine Architecture</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>Acoustic ResNet-18 (Fold-4 Val: 88.75% · Fold-5 Test: 80.00% · Macro-F1: 0.787)</p>
          </div>
          <button className="secondary-btn" onClick={onClose}><X size={14} /> Close</button>
        </div>
        <p className="muted" style={{ marginTop: 0 }}>
          End-to-End Pipeline: Spectral Subtraction OLA denoising, Autocorrelation F0 pitch tracking, Levinson-Durbin LPC vocal tract formants,
          K-Means VQ speaker diarization, Subsequence DTW query spotter, Silero VAD, Discrete Markov Scene &amp; Prosody HMM smoothing, Groq Whisper + LLaMA-3 Copilot.
        </p>
        <table className="data-table">
          <thead><tr><th>ESC-50 Category</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r[0]}><td><code>{r[0]}</code></td><td className="mono">{r[1]}</td><td className="mono">{r[2]}</td><td className="mono"><b>{r[3]}</b></td><td className="mono">{r[4]}</td></tr>
            ))}
          </tbody>
        </table>
        <div style={{ marginTop: 14, fontSize: 12 }} className="tag ok"><CheckCircle2 size={13} /> Weights Loaded: backend/src/soniccaption/services/weights/</div>
      </div>
    </div>
  )
}
