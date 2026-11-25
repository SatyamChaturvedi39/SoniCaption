import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertCircle,
  AudioLines,
  Bot,
  ChevronDown,
  ChevronUp,
  Download,
  FileAudio,
  Info,
  Mic,
  Radio,
  ShieldCheck,
  Square,
  Upload,
} from 'lucide-react'
import { analyzeAudio, exportResult, getHealth, getSimilarSessions, queryAudioByVoice } from './api'
import { AgentView, EngineSpecsModal, VoiceAuthenticatorView, WatchView } from './components/StudioTabs'
import { DualLineCanvas, SpectrogramCanvas } from './components/VisualCanvas'
import { WavePlayer } from './components/WavePlayer'
import type { Analysis, DtwSpotMatch } from './types'
import { RAW_MIC_CONSTRAINTS, blobToWavFile } from './utils/wavEncoder'

type Tab = 'workspace' | 'authenticator' | 'watch' | 'agent'

const STAGES = [
  'Running Spectral Subtraction DSP...',
  'Detecting speech boundaries (Silero VAD)...',
  'Transcribing speech, prosody & speakers (VQ)...',
  'Classifying 10 environmental sound classes...',
  'Decoding scene/prosody HMMs & DTW matches...',
]

export default function App() {
  const [tab, setTab] = useState<Tab>('workspace')
  const [inputMode, setInputMode] = useState<'record' | 'upload'>('upload')
  const [denoise, setDenoise] = useState(true)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [processing, setProcessing] = useState(false)
  const [stage, setStage] = useState(0)
  const [error, setError] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [audioUrl, setAudioUrl] = useState('')
  const [result, setResult] = useState<Analysis | null>(null)
  const [health, setHealth] = useState<any>(null)
  const [seekTo, setSeekTo] = useState<{ time: number; seq: number } | null>(null)
  const [showSpecs, setShowSpecs] = useState(false)
  const [drag, setDrag] = useState(false)

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const reloadMeta = () => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }

  useEffect(() => { reloadMeta() }, [])
  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t: MediaStreamTrack) => t.stop())
    if (audioUrl) URL.revokeObjectURL(audioUrl)
  }, [audioUrl])

  function chooseFile(f?: File) {
    if (!f) return
    setError('')
    setFile(f)
    setResult(null)
    setAudioUrl(URL.createObjectURL(f))
  }

  async function startRecording() {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError('Microphone recording is not supported in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW_MIC_CONSTRAINTS })
      streamRef.current = stream
      const rec = new MediaRecorder(stream)
      mediaRecorder.current = rec
      chunks.current = []
      rec.ondataavailable = (e) => { if (e.data.size) chunks.current.push(e.data) }
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        const blob = new Blob(chunks.current, { type: rec.mimeType || 'audio/webm' })
        if (!blob.size) {
          setError('No audio captured. Check microphone level and try again.')
          return
        }
        try {
          const wavFile = await blobToWavFile(blob, 'mic-capture')
          setFile(wavFile)
          setAudioUrl(URL.createObjectURL(wavFile))
        } catch {
          const rawFile = new File([blob], `mic-capture-${Date.now()}.webm`, { type: blob.type })
          setFile(rawFile)
          setAudioUrl(URL.createObjectURL(blob))
        }
      }
      rec.start()
      setRecording(true)
      setSeconds(0)
      timer.current = window.setInterval(() => setSeconds((s) => s + 1), 1000)
    } catch {
      setError('Microphone permission denied or unavailable.')
    }
  }

  function stopRecording() {
    if (timer.current) clearInterval(timer.current)
    if (mediaRecorder.current?.state === 'recording') mediaRecorder.current.stop()
    setRecording(false)
  }

  async function processAudio() {
    if (!file) return
    setError('')
    setProcessing(true)
    setStage(0)
    const tick = window.setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 900)
    try {
      const data = await analyzeAudio(file, {
        denoise,
        sourceType: inputMode === 'record' ? 'mic_recording' : 'upload',
      })
      setResult(data)
      setTab('workspace')
      reloadMeta()
    } catch (e: any) {
      setError(e.message || 'Unable to analyze audio')
    } finally {
      clearInterval(tick)
      setProcessing(false)
    }
  }

  const fmt = (s: number) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${Math.floor(s % 60).toString().padStart(2, '0')}`
  const isLive = health?.mode === 'live'
  const nav: [Tab, string, any][] = [
    ['workspace', 'Audio Studio', AudioLines],
    ['authenticator', 'AI Voice Authenticator', ShieldCheck],
    ['watch', 'Live Room Radar', Radio],
    ['agent', 'Voice Copilot', Bot],
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><AudioLines size={18} /></div>
          <div><b>SonicVoice Studio</b><small>Acoustic Intelligence</small></div>
        </div>

        <div className="nav-label">Workspace</div>
        {nav.map(([id, label, Icon]) => (
          <button key={id} className={`nav-item ${tab === id ? 'active' : ''}`} onClick={() => { setTab(id); setError('') }}>
            <Icon size={16} />
            <span>{label}</span>
          </button>
        ))}

        <div className="sidebar-bottom">
          <button className="secondary-btn" style={{ width: '100%', justifyContent: 'center' }} onClick={() => setShowSpecs(true)}>
            <Info size={14} /> Engine Architecture
          </button>
          <div className="status-line">
            <span className={`status-dot ${isLive ? 'live' : ''}`} />
            {isLive ? 'Backend Live' : 'Backend Offline'}
          </div>
          <div className="mono" style={{ fontSize: 10, color: '#64748b' }}>
            ASR: {health?.groq_configured ? 'Groq Whisper v3' : 'Silero VAD fallback'}
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{nav.find((n) => n[0] === tab)?.[1]}</h1>
            <span className="topbar-sub">Prosody-Aware Transcription, Spoken DTW Retrieval &amp; Ambient Event Intelligence</span>
          </div>
          <div className="topbar-meta">
            <span className={`tag ${isLive ? 'ok' : ''}`}>{isLive ? 'Models + VAD Ready' : 'Mock Fallback'}</span>
            <span className={`tag ${health?.groq_configured ? 'ok' : 'warn'}`}>
              {health?.groq_configured ? 'GROQ_API_KEY Loaded (.env)' : 'Set GROQ_API_KEY in .env'}
            </span>
          </div>
        </header>

        {error && <div className="alert-bar"><AlertCircle size={16} />{error}<button onClick={() => setError('')}>Dismiss</button></div>}

        {tab === 'workspace' && (
          <div className="content">
            {!result ? (
              <>
                <div className="metric-strip">
                  <div className="metric-cell"><small>Acoustic Engine</small><b>SonicAcousticResNet-18</b></div>
                  <div className="metric-cell"><small>Speech Engine</small><b>{health?.groq_configured ? 'Groq Whisper v3' : 'VAD Fallback'}</b></div>
                  <div className="metric-cell"><small>DSP Pipeline</small><b>Spectral OLA · LPC · DTW · HMM</b></div>
                  <div className="metric-cell"><small>Acoustic Analysis</small><b>F0 Pitch &amp; Prosody</b></div>
                </div>

                <div className="panel">
                  <div className="panel-header">
                    <div className="seg-tabs">
                      <button className={inputMode === 'upload' ? 'selected' : ''} onClick={() => { setInputMode('upload'); setFile(null); setAudioUrl('') }}><Upload size={14} /> Upload Audio</button>
                      <button className={inputMode === 'record' ? 'selected' : ''} onClick={() => { setInputMode('record'); setFile(null); setAudioUrl('') }}><Mic size={14} /> Record Microphone</button>
                    </div>
                    <label className="toggle-chip">
                      <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
                      Spectral Subtraction Denoise
                    </label>
                  </div>

                  {inputMode === 'record' ? (
                    <div className="record-box">
                      <div className={`mic-badge ${recording ? 'live' : ''}`}><Mic size={22} /></div>
                      <h3>{recording ? 'Recording Audio...' : file ? 'Recording Ready (16 kHz WAV)' : 'Capture Speech & Ambient Events'}</h3>
                      <p className="muted" style={{ margin: '4px 0 0' }}>Speak naturally or produce foreground sounds (clapping, knocking, laughter). Recorded at raw 16 kHz PCM.</p>
                      <div className="record-clock">{fmt(seconds)}</div>
                      {!recording ? (
                        <button className="primary-btn" onClick={startRecording}><Mic size={15} /> Start Recording</button>
                      ) : (
                        <button className="danger-btn" onClick={stopRecording}><Square size={14} /> Stop Recording</button>
                      )}
                    </div>
                  ) : (
                    <div
                      className={`dropzone ${drag ? 'dragging' : ''}`}
                      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
                      onDragLeave={() => setDrag(false)}
                      onDrop={(e) => { e.preventDefault(); setDrag(false); chooseFile(e.dataTransfer.files[0]) }}
                    >
                      <div className="mic-badge"><Upload size={20} /></div>
                      <h3>Drop audio file or select from disk</h3>
                      <p className="muted" style={{ margin: '4px 0 12px' }}>Supports WAV, MP3, OGG, M4A, FLAC, and WebM.</p>
                      <label className="secondary-btn">
                        Choose File
                        <input type="file" accept="audio/*" hidden onChange={(e) => chooseFile(e.target.files?.[0])} />
                      </label>
                    </div>
                  )}

                  {file && (
                    <div className="list-card" style={{ marginTop: 14 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <FileAudio size={18} color="#06b6d4" />
                        <div><b>{file.name}</b> <span className="muted mono">({(file.size / 1024).toFixed(1)} KB)</span></div>
                      </div>
                      <button className="secondary-btn" onClick={() => { setFile(null); setAudioUrl('') }}>Clear</button>
                    </div>
                  )}

                  {audioUrl && <div style={{ marginTop: 14 }}><WavePlayer audioUrl={audioUrl} /></div>}

                  {file && (
                    <div className="row-between" style={{ marginTop: 16 }}>
                      <span className="muted mono" style={{ fontSize: 12 }}>{processing ? STAGES[stage] : 'Runs denoising, VAD, Whisper ASR, prosody & speaker VQ, HMM smoothing & sound model.'}</span>
                      <button className="primary-btn" onClick={processAudio} disabled={processing}>
                        {processing ? 'Running Pipeline...' : 'Analyze Audio'}
                      </button>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <ResultsView
                data={result}
                audioUrl={audioUrl}
                seekTo={seekTo}
                onSeek={(t) => setSeekTo((prev) => ({ time: t, seq: (prev?.seq || 0) + 1 }))}
                onOpenSimilar={async (sessionId) => {
                  const sRes = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/v1/sessions`)
                  if (!sRes.ok) return
                  const { sessions } = await sRes.json()
                  const match = sessions.find((x: any) => x.id === sessionId)
                  if (!match) return
                  const sim = await getSimilarSessions(match.id).catch(() => ({ matches: [] }))
                  setAudioUrl('')
                  setResult({
                    session_id: match.id,
                    filename: match.filename,
                    duration: match.duration,
                    sample_rate: 16000,
                    mode: 'live',
                    summary: match.summary,
                    similar_sessions: sim.matches,
                    speech_segments: match.timeline.filter((t: any) => t.category === 'speech'),
                    sound_events: match.timeline.filter((t: any) => t.category === 'sound_event'),
                    timeline: match.timeline,
                    metrics: match.metrics,
                  })
                }}
                onBack={() => setResult(null)}
              />
            )}
          </div>
        )}

        {tab === 'authenticator' && <VoiceAuthenticatorView />}
        {tab === 'watch' && <WatchView onAlertLogged={reloadMeta} />}
        {tab === 'agent' && <AgentView />}
      </main>

      {showSpecs && <EngineSpecsModal onClose={() => setShowSpecs(false)} />}
    </div>
  )
}

function ResultsView({
  data,
  audioUrl,
  seekTo,
  onSeek,
  onOpenSimilar,
  onBack,
}: {
  data: Analysis
  audioUrl: string
  seekTo: { time: number; seq: number } | null
  onSeek: (t: number) => void
  onOpenSimilar: (sessionId: number) => void
  onBack: () => void
}) {
  const [showDsp, setShowDsp] = useState(false)
  const [spotRec, setSpotRec] = useState(false)
  const [spotBusy, setSpotBusy] = useState(false)
  const [spotErr, setSpotErr] = useState('')
  const [spotMatches, setSpotMatches] = useState<DtwSpotMatch[] | null>(null)
  const spotRecRef = useRef<MediaRecorder | null>(null)
  const spotChunksRef = useRef<BlobPart[]>([])
  const spotStreamRef = useRef<MediaStream | null>(null)

  useEffect(() => () => spotStreamRef.current?.getTracks().forEach((t) => t.stop()), [])

  async function runSpotQuery(queryFile: File) {
    setSpotBusy(true)
    setSpotErr('')
    try {
      const res = await queryAudioByVoice(queryFile, Boolean(data.metrics?.denoised ?? true))
      setSpotMatches(res.matches)
    } catch (e: any) {
      setSpotErr(e.message || 'DTW query spotter requires an active analyzed clip.')
    } finally {
      setSpotBusy(false)
    }
  }

  async function toggleSpotMic() {
    if (spotRec) {
      spotRecRef.current?.stop()
      setSpotRec(false)
      return
    }
    setSpotErr('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW_MIC_CONSTRAINTS })
      spotStreamRef.current = stream
      const rec = new MediaRecorder(stream)
      spotRecRef.current = rec
      spotChunksRef.current = []
      rec.ondataavailable = (e) => { if (e.data.size) spotChunksRef.current.push(e.data) }
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(spotChunksRef.current, { type: rec.mimeType || 'audio/webm' })
        if (!blob.size) return
        const wav = await blobToWavFile(blob, 'dtw-query')
        await runSpotQuery(wav)
      }
      rec.start()
      setSpotRec(true)
    } catch {
      setSpotErr('Microphone access needed to speak a search phrase.')
    }
  }

  const warnings: string[] = data.metrics?.warnings || []
  const dropped: any[] = data.metrics?.dropped_events || []
  const highlights = data.reminders || []
  const sceneHmm = data.metrics?.scene_hmm
  const prosodyHmm = data.metrics?.prosody_hmm
  const diag = data.metrics?.diagnostics
  const similar = data.similar_sessions || data.metrics?.similar_sessions || []

  return (
    <div>
      <div className="row-between" style={{ marginBottom: 16 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 19 }}>{data.filename}</h2>
          <span className="muted mono">Log #{data.session_id ?? '—'} · {data.duration.toFixed(2)}s · {data.metrics?.model_architecture || 'ResNet-18'} · ASR: {data.metrics?.speech_source || 'live'}</span>
        </div>
        <button className="secondary-btn" onClick={onBack}>← Back to Input</button>
      </div>

      <div className="metric-strip">
        <div className="metric-cell"><small>Speech Turns</small><b>{data.speech_segments.length}</b></div>
        <div className="metric-cell"><small>Sound Events</small><b>{data.sound_events.length}</b></div>
        <div className="metric-cell"><small>SNR Gain</small><b>{data.metrics?.denoised ? `+${data.metrics?.snr_gain_db ?? 0} dB` : 'Off'}</b></div>
        <div className="metric-cell"><small>Mean Pitch (F0)</small><b>{data.metrics?.mean_f0_hz ? `${data.metrics.mean_f0_hz} Hz` : '—'}</b></div>
      </div>

      {warnings.length > 0 && <div className="notice-bar">{warnings.join(' ')}</div>}

      {data.metrics?.voice_authenticity && data.metrics.voice_authenticity.verdict && (
        <section
          className="panel"
          style={{
            background: data.metrics.voice_authenticity.is_synthetic
              ? 'rgba(239, 68, 68, 0.08)'
              : data.metrics.voice_authenticity.authenticity_score >= 68
              ? 'rgba(16, 185, 129, 0.08)'
              : 'rgba(245, 158, 11, 0.08)',
            borderColor: data.metrics.voice_authenticity.is_synthetic
              ? 'rgba(239, 68, 68, 0.4)'
              : data.metrics.voice_authenticity.authenticity_score >= 68
              ? 'rgba(16, 185, 129, 0.4)'
              : 'rgba(245, 158, 11, 0.4)',
          }}
        >
          <div className="row-between">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <ShieldCheck size={20} color={data.metrics.voice_authenticity.is_synthetic ? '#ef4444' : '#10b981'} />
              <div>
                <b>Voice Authenticity: {data.metrics.voice_authenticity.verdict}</b>
                <div className="muted mono">
                  Authenticity: {data.metrics.voice_authenticity.authenticity_score}% · Jitter: {data.metrics.voice_authenticity.metrics?.jitter_percent}% · Shimmer: {data.metrics.voice_authenticity.metrics?.shimmer_db} dB
                </div>
              </div>
            </div>
            <span className={`tag ${data.metrics.voice_authenticity.is_synthetic ? 'warn' : 'ok'}`}>
              {data.metrics.voice_authenticity.is_synthetic ? 'AI Vocoder Signature' : 'Natural Human Vocal Tract'}
            </span>
          </div>
        </section>
      )}

      {data.summary && (
        <section className="panel">
          <div className="panel-header"><h3>Multimodal Audio Synthesis &amp; Key Spoken Highlights</h3></div>
          <p style={{ margin: '0 0 14px', fontSize: 13.5, lineHeight: 1.6, color: '#e2e8f0' }}>{data.summary}</p>
          {highlights.length > 0 ? (
            highlights.map((r, i) => (
              <div key={i} className="reminder-row">
                <div><b>{r.title}</b> <span className="sound-chip">{r.due_time}</span></div>
                <span className="tag">{r.trigger_source}</span>
              </div>
            ))
          ) : null}
        </section>
      )}

      {audioUrl && (
        <section className="panel">
          <h3 style={{ marginBottom: 12 }}>Waveform Scrub Player</h3>
          <WavePlayer audioUrl={audioUrl} seekTo={seekTo} />
        </section>
      )}

      <section className="panel">
        <div className="panel-header">
          <div>
            <h3>Chronological Caption &amp; Prosody Timeline</h3>
            <p className="muted" style={{ margin: '2px 0 0' }}>Click any row to seek playback. Speech rows include vocal pitch &amp; prosody delivery tags.</p>
          </div>
          <div className="export-bar">
            <Download size={14} color="#94a3b8" />
            {['srt', 'vtt', 'json', 'csv', 'txt'].map((f) => (
              <button key={f} onClick={() => exportResult(f, data).catch(() => alert('Export failed'))}>
                .{f}
              </button>
            ))}
          </div>
        </div>
        {data.timeline.length === 0 ? (
          <p className="muted">No speech or foreground events exceeded calibrated room thresholds.</p>
        ) : (
          <div className="timeline">
            {data.timeline.map((item) => (
              <button key={item.id} className={`timeline-row ${item.category}`} onClick={() => onSeek(item.start)}>
                <span className="time">{item.start.toFixed(2)}s – {item.end.toFixed(2)}s</span>
                <span className="actor">{item.category === 'speech' ? (item.speaker || item.label || 'Speaker 1') : item.label}</span>
                <span className="body">
                  {item.display_text}
                  {item.category === 'speech' && item.delivery_style && (
                    <span className="sound-chip" style={{ marginLeft: 8 }}>
                      {item.delivery_style}
                      {item.mean_f0_hz ? ` · ${item.mean_f0_hz} Hz` : ''}
                      {item.speaking_rate_wps ? ` · ${item.speaking_rate_wps} w/s` : ''}
                    </span>
                  )}
                </span>
                <span className="conf">{item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : ''}</span>
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h3>Query-by-Voice Acoustic Spotter (Subsequence DTW)</h3>
            <p className="muted" style={{ margin: '2px 0 0' }}>Speak a word, name, or sound to locate its closest acoustic MFCC matches inside this recording—works even without text matching.</p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className={spotRec ? 'danger-btn' : 'primary-btn'} onClick={toggleSpotMic} disabled={spotBusy}>
              {spotRec ? <Square size={14} /> : <Mic size={14} />}
              {spotRec ? 'Stop & Spot' : spotBusy ? 'Warping MFCCs...' : 'Speak Query'}
            </button>
            <label className="secondary-btn">
              Upload Query
              <input type="file" accept="audio/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) runSpotQuery(f) }} />
            </label>
          </div>
        </div>
        {spotErr && <div className="notice-bar">{spotErr}</div>}
        {spotMatches && (
          spotMatches.length === 0 ? (
            <p className="muted">No distinct active matches found for that query.</p>
          ) : (
            spotMatches.map((m, i) => (
              <div key={i} className="list-card">
                <div>
                  <b>Window {m.start.toFixed(2)}s – {m.end.toFixed(2)}s</b>{' '}
                  <span className="tag ok mono">{m.similarity_pct}% MFCC similarity</span>
                  <div className="muted mono">Subsequence DTW cost: {m.normalized_distance}</div>
                </div>
                <button className="secondary-btn" onClick={() => onSeek(m.start)}>Jump to {m.start.toFixed(1)}s</button>
              </div>
            ))
          )
        )}
      </section>

      {(sceneHmm?.best_path?.length > 0 || prosodyHmm?.best_path?.length > 0) && (
        <section className="panel">
          {sceneHmm?.best_path?.length > 0 && (
            <div style={{ marginBottom: prosodyHmm?.best_path?.length > 0 ? 16 : 0 }}>
              <div className="panel-header" style={{ marginBottom: 8 }}>
                <div>
                  <h3>Acoustic Scene Flow (Markov Viterbi)</h3>
                  <p className="muted" style={{ margin: '2px 0 0' }}>Scene states per {sceneHmm.step_s}s window (log P = {sceneHmm.log_likelihood?.toFixed(2)}).</p>
                </div>
              </div>
              <div className="tag-row">
                {sceneHmm.best_path.map((st: string, i: number) => (
                  <span key={i} className={`tag mono ${st === 'Speech' ? 'ok' : st === 'SoundEvent' ? 'warn' : ''}`}>
                    {i * sceneHmm.step_s}s: {st}
                  </span>
                ))}
              </div>
            </div>
          )}
          {prosodyHmm?.best_path?.length > 0 && data.speech_segments.length > 0 && (
            <div>
              <div className="panel-header" style={{ marginBottom: 8 }}>
                <div>
                  <h3>Discourse &amp; Vocal Emphasis Flow (Prosody HMM)</h3>
                  <p className="muted" style={{ margin: '2px 0 0' }}>Decoded conversation states (Explanation, KeyEmphasis, QuestionPrompt, TurnPause) from F0 + RMS cues.</p>
                </div>
              </div>
              <div className="tag-row">
                {prosodyHmm.best_path.map((st: string, i: number) => (
                  <span key={i} className={`tag mono ${st === 'KeyEmphasis' ? 'warn' : st === 'QuestionPrompt' || st === 'Explanation' ? 'ok' : ''}`}>
                    {i * prosodyHmm.step_s}s: {st}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {similar.length > 0 && (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>Acoustically Similar Past Recordings (DTW Match)</h3>
              <p className="muted" style={{ margin: '2px 0 0' }}>Ranked by MFCC Dynamic Time Warping distance against your audio logs.</p>
            </div>
          </div>
          {similar.map((m: any) => (
            <div key={m.session_id} className="list-card">
              <div>
                <b>{m.filename}</b> <span className="tag ok mono">{m.similarity_pct}% match</span>
                <div className="muted mono">Normalized DTW cost: {m.normalized_distance} · {m.created_at}</div>
              </div>
              <button className="secondary-btn" onClick={() => onOpenSimilar(m.session_id)}>Open Log</button>
            </div>
          ))}
        </section>
      )}

      {diag && (
        <section className="panel">
          <div className="panel-header" style={{ marginBottom: showDsp ? 14 : 0 }}>
            <div>
              <h3><Activity size={15} style={{ verticalAlign: 'middle', marginRight: 6, color: '#06b6d4' }} />Audio Signal &amp; Formant Diagnostics</h3>
              <p className="muted" style={{ margin: '2px 0 0' }}>
                SNR: {diag.snr_before_db} dB → {diag.snr_after_db} dB · LPC Formants: {diag.lpc_formants_hz?.slice(0, 3).join(', ') || '—'} Hz
              </p>
            </div>
            <button className="secondary-btn" onClick={() => setShowDsp((v) => !v)}>
              {showDsp ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {showDsp ? 'Hide Plots' : 'Inspect Plots'}
            </button>
          </div>
          {showDsp && (
            <div className="grid-2">
              <DualLineCanvas primary={diag.waveform || []} secondary={diag.waveform_denoised || []} labelA="Raw Audio" labelB="Denoised (Boll OLA)" />
              <SpectrogramCanvas matrix={diag.stft_matrix || []} title="2D STFT Power Spectrogram (dB)" />
              <SpectrogramCanvas matrix={diag.mfcc || []} title="13-MFCC Cepstral Matrix" />
              <DualLineCanvas primary={diag.fft_mag_db || []} secondary={diag.lpc_envelope_db || []} labelA="FFT Spectrum (dB)" labelB="LPC Formant Envelope (dB)" />
            </div>
          )}
        </section>
      )}

      {dropped.length > 0 && (
        <section className="panel">
          <div className="panel-header"><h3>Filtered / Sub-Threshold Ambient Detections ({dropped.length})</h3></div>
          {dropped.slice(0, 12).map((d, i) => (
            <div key={i} className="list-card">
              <div>
                <b>{d.item?.label || 'Segment'}</b> {d.item?.confidence != null ? <span className="mono">({(d.item.confidence * 100).toFixed(1)}%)</span> : ''}
                <div className="muted">{d.reason}</div>
              </div>
              <span className="muted mono">{d.item?.start != null ? `${d.item.start.toFixed(2)}s` : ''}</span>
            </div>
          ))}
        </section>
      )}
    </div>
  )
}
