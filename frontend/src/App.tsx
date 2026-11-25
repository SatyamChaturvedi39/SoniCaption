import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertCircle,
  AudioLines,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Download,
  FileAudio,
  GitBranch,
  History,
  KeyRound,
  Mic,
  Network,
  Radio,
  Sparkles,
  Square,
  Upload,
} from 'lucide-react'
import { analyzeAudio, exportResult, getHealth, getReminders, setGroqKey } from './api'
import { AgentView, HistoryAndRemindersView, LabsAndModelView, WatchView } from './components/StudioTabs'
import { WavePlayer } from './components/WavePlayer'
import type { Analysis } from './types'
import { blobToWavFile } from './utils/wavEncoder'

type Tab = 'workspace' | 'agent' | 'watch' | 'history' | 'signal' | 'pattern' | 'hmm' | 'model'

const STAGES = [
  'Applying Member 1 Spectral Subtraction DSP',
  'Detecting speech boundaries (Silero VAD)',
  'Running Whisper ASR & Speaker VQ Diarization',
  'Classifying 10 ambient ESC-50 sounds (ResNet-18)',
  'Fusing timeline & extracting AI reminders',
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
  const [pendingCount, setPendingCount] = useState(0)
  const [refreshKey, setRefreshKey] = useState(0)
  const [seekTo, setSeekTo] = useState<{ time: number; seq: number } | null>(null)
  const [showKeyModal, setShowKeyModal] = useState(false)
  const [groqInput, setGroqInput] = useState('')
  const [drag, setDrag] = useState(false)

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const reloadMeta = () => {
    getHealth().then(setHealth).catch(() => setHealth(null))
    getReminders()
      .then((r) => setPendingCount(r.reminders.filter((x) => x.status === 'pending').length))
      .catch(() => {})
    setRefreshKey((k) => k + 1)
  }

  useEffect(() => { reloadMeta() }, [])
  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
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
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
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
          setError('No audio was captured. Please try recording again.')
          return
        }
        try {
          const wavFile = await blobToWavFile(blob, 'studio-mic')
          setFile(wavFile)
          setAudioUrl(URL.createObjectURL(wavFile))
        } catch {
          const rawFile = new File([blob], `studio-mic-${Date.now()}.webm`, { type: blob.type })
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
  const nav: [Tab, string, any, number?][] = [
    ['workspace', 'Audio Studio', AudioLines],
    ['agent', 'Voice Copilot', Bot],
    ['watch', 'Live Watch', Radio],
    ['history', 'Logs & Reminders', History, pendingCount],
    ['signal', 'Signal & DSP Lab', Activity],
    ['pattern', 'Pattern Lab (DTW)', GitBranch],
    ['hmm', 'HMM Trellis', Network],
    ['model', 'Model Card', BrainCircuit],
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><AudioLines size={22} /></div>
          <div><b>SonicCaption</b><small>Acoustic Copilot Studio</small></div>
        </div>
        <div className="nav-label">PRODUCT WORKSPACE</div>
        {nav.slice(0, 4).map(([id, label, Icon, badge]) => (
          <button key={id} className={`nav-item ${tab === id ? 'active' : ''}`} onClick={() => { setTab(id); setError('') }}>
            <Icon size={17} />
            <span>{label}</span>
            {badge ? <em className="nav-badge">{badge}</em> : null}
          </button>
        ))}
        <div className="nav-label">SPEECH PROCESSING LABS</div>
        {nav.slice(4).map(([id, label, Icon]) => (
          <button key={id} className={`nav-item ${tab === id ? 'active' : ''}`} onClick={() => { setTab(id); setError('') }}>
            <Icon size={17} />
            <span>{label}</span>
          </button>
        ))}
        <div className="sidebar-bottom">
          <div><span className={`status-dot ${isLive ? 'live' : ''}`} />{isLive ? 'LIVE ML PIPELINE' : 'BACKEND OFFLINE'}</div>
          <div>ASR: {health?.groq_configured ? 'Groq Whisper Large v3' : 'Silero VAD (Key Optional)'}</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <span className="eyebrow">SONICCAPTION · MULTIMODAL ACOUSTIC INTELLIGENCE</span>
            <h1>{nav.find((n) => n[0] === tab)?.[1]}</h1>
          </div>
          <div className="topbar-actions">
            <span className={`pill ${isLive ? 'live' : ''}`}>● {isLive ? 'ResNet-18 + VAD + DSP Active' : 'Offline / Mock'}</span>
            <button className={`pill ${health?.groq_configured ? 'live' : 'warn'}`} onClick={() => setShowKeyModal(true)}>
              <KeyRound size={13} />
              {health?.groq_configured ? 'Groq Cloud Ready' : 'Set Groq Key'}
            </button>
          </div>
        </header>

        {error && <div className="alert"><AlertCircle size={17} />{error}<button onClick={() => setError('')}>Dismiss</button></div>}

        {tab === 'workspace' && (
          <div className="content">
            {!result ? (
              <>
                <section className="hero">
                  <div>
                    <span className="eyebrow">DUAL-STREAM AUDIO COPILOT</span>
                    <h2>Hear words. Understand <em>everything else.</em></h2>
                    <p>Record directly from your microphone or upload audio. SonicCaption applies spectral subtraction denoising, speaker diarization, Whisper ASR, 10-class ResNet-18 acoustic sensing, and extracts reminders automatically.</p>
                  </div>
                  <div className="hero-badge-col">
                    <div className="hero-stat">ESC-50 Sound Model<b>88.75% Val / 80.0% Test</b></div>
                    <div className="hero-stat">Active Persistent Logs<b>{pendingCount} Pending Reminders</b></div>
                  </div>
                </section>

                <div className="row-between" style={{ marginBottom: 12 }}>
                  <div className="tabs" style={{ marginBottom: 0 }}>
                    <button className={inputMode === 'upload' ? 'selected' : ''} onClick={() => { setInputMode('upload'); setFile(null); setAudioUrl('') }}><Upload size={16} /> Upload Audio</button>
                    <button className={inputMode === 'record' ? 'selected' : ''} onClick={() => { setInputMode('record'); setFile(null); setAudioUrl('') }}><Mic size={16} /> Live Mic Recording</button>
                  </div>
                  <label className="toggle-chip">
                    <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
                    Spectral Subtraction Denoising
                  </label>
                </div>

                <section className="panel">
                  {inputMode === 'record' ? (
                    <div className="record-area">
                      <div className={`mic-orb ${recording ? 'is-recording' : ''}`}><Mic size={30} /></div>
                      <h3>{recording ? 'Listening to Microphone...' : file ? 'Recording Converted to 16 kHz WAV' : 'Record Voice & Background Audio'}</h3>
                      <p className="muted">{recording ? 'Speak or make background sounds (clapping, door knock, coughing, laughing).' : 'Automatically converts browser audio cleanly to 16 kHz mono WAV for live analysis.'}</p>
                      <div className="record-time">{fmt(seconds)}</div>
                      {!recording ? (
                        <button className="primary-btn" onClick={startRecording}><Mic size={16} /> Start Recording</button>
                      ) : (
                        <button className="danger-btn" onClick={stopRecording}><Square size={15} /> Stop Recording</button>
                      )}
                    </div>
                  ) : (
                    <div
                      className={`dropzone ${drag ? 'dragging' : ''}`}
                      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
                      onDragLeave={() => setDrag(false)}
                      onDrop={(e) => { e.preventDefault(); setDrag(false); chooseFile(e.dataTransfer.files[0]) }}
                    >
                      <div className="mic-orb"><Upload size={26} /></div>
                      <h3>Drop WAV, MP3, OGG, M4A, FLAC or WebM here</h3>
                      <p className="muted">Try speaking with background clapping, coughing, door knock, rain, siren, or laughing.</p>
                      <label className="secondary-btn" style={{ marginTop: 10 }}>
                        Browse Audio File
                        <input type="file" accept="audio/*" hidden onChange={(e) => chooseFile(e.target.files?.[0])} />
                      </label>
                    </div>
                  )}

                  {file && (
                    <div className="list-card" style={{ marginTop: 16 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <FileAudio size={18} color="#38bdf8" />
                        <div><b>{file.name}</b><div className="muted">{(file.size / 1024).toFixed(1)} KB</div></div>
                      </div>
                      <button className="secondary-btn" onClick={() => { setFile(null); setAudioUrl('') }}>Remove</button>
                    </div>
                  )}

                  {audioUrl && <div style={{ marginTop: 14 }}><WavePlayer audioUrl={audioUrl} /></div>}

                  {file && (
                    <div className="row-between" style={{ marginTop: 16 }}>
                      <span className="muted">{processing ? STAGES[stage] : 'Ready to run full VAD + Whisper + ResNet-18 + Fusion + Agent insights.'}</span>
                      <button className="primary-btn" onClick={processAudio} disabled={processing}>
                        <Sparkles size={15} /> {processing ? 'Analyzing...' : 'Analyze Audio'}
                      </button>
                    </div>
                  )}
                </section>
              </>
            ) : (
              <ResultsView
                data={result}
                audioUrl={audioUrl}
                seekTo={seekTo}
                onSeek={(t) => setSeekTo((prev) => ({ time: t, seq: (prev?.seq || 0) + 1 }))}
                onBack={() => setResult(null)}
              />
            )}
          </div>
        )}

        {tab === 'agent' && <AgentView onRemindersChanged={reloadMeta} />}
        {tab === 'watch' && <WatchView onAlertLogged={reloadMeta} />}
        {tab === 'history' && (
          <HistoryAndRemindersView
            refreshKey={refreshKey}
            onOpenSession={(sess) => {
              setResult(sess)
              setAudioUrl('')
              setTab('workspace')
            }}
          />
        )}
        {tab === 'signal' && <LabsAndModelView kind="signal" />}
        {tab === 'pattern' && <LabsAndModelView kind="pattern" />}
        {tab === 'hmm' && <LabsAndModelView kind="hmm" />}
        {tab === 'model' && <LabsAndModelView kind="model" />}
      </main>

      {showKeyModal && (
        <div className="modal-backdrop" onClick={() => setShowKeyModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>Configure Free Groq API Key</h3>
            <p className="muted">Stored only in backend memory for this running session. Unlocks Whisper Large v3 speech transcription and Llama-3.3-70B Voice Copilot reasoning.</p>
            <input
              className="input-field"
              type="password"
              style={{ width: '100%', marginTop: 10 }}
              placeholder="gsk_..."
              value={groqInput}
              onChange={(e) => setGroqInput(e.target.value)}
            />
            <div className="row-between" style={{ marginTop: 16 }}>
              <button className="secondary-btn" onClick={() => setShowKeyModal(false)}>Cancel</button>
              <button
                className="primary-btn"
                onClick={async () => {
                  await setGroqKey(groqInput)
                  setShowKeyModal(false)
                  setGroqInput('')
                  reloadMeta()
                }}
              >
                <CheckCircle2 size={14} /> Save Session Key
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ResultsView({
  data,
  audioUrl,
  seekTo,
  onSeek,
  onBack,
}: {
  data: Analysis
  audioUrl: string
  seekTo: { time: number; seq: number } | null
  onSeek: (t: number) => void
  onBack: () => void
}) {
  const warnings: string[] = data.metrics?.warnings || []
  const dropped: any[] = data.metrics?.dropped_events || []
  const reminders = data.reminders || []

  return (
    <div>
      <div className="row-between" style={{ marginBottom: 16 }}>
        <div>
          <span className="eyebrow">ANALYSIS SAVED TO CALL LOGS #{data.session_id ?? '—'}</span>
          <h2 style={{ margin: '4px 0' }}>{data.filename}</h2>
          <span className="muted">{data.duration.toFixed(1)}s · {data.sample_rate} Hz · ASR: {data.metrics?.speech_source || 'live'}</span>
        </div>
        <button className="secondary-btn" onClick={onBack}>← New Analysis</button>
      </div>

      <div className="metric-grid">
        <div className="metric"><small>Speech Turns</small><b>{data.speech_segments.length}</b></div>
        <div className="metric"><small>Sound Events</small><b>{data.sound_events.length}</b></div>
        <div className="metric"><small>SNR Gain (Denoised)</small><b>{data.metrics?.denoised ? `+${data.metrics?.snr_gain_db ?? 0} dB` : 'Bypassed'}</b></div>
        <div className="metric"><small>Mean Pitch (F0)</small><b>{data.metrics?.mean_f0_hz ? `${data.metrics.mean_f0_hz} Hz` : '—'}</b></div>
      </div>

      {warnings.length > 0 && <div className="banner-warn">{warnings.join(' ')}</div>}

      {data.summary && (
        <section className="panel">
          <h3><Sparkles size={15} style={{ verticalAlign: 'middle', marginRight: 6 }} />AI Session Summary & Auto-Extracted Reminders</h3>
          <p style={{ margin: '0 0 10px', fontSize: 13, lineHeight: 1.6 }}>{data.summary}</p>
          {reminders.length > 0 ? (
            reminders.map((r, i) => (
              <div key={i} className="reminder-item">
                <div><b>{r.title}</b> <span className="sound-chip">{r.due_time}</span></div>
                <span className="pill">{r.trigger_source}</span>
              </div>
            ))
          ) : (
            <span className="muted">No explicit reminders or urgent acoustic triggers found in this clip.</span>
          )}
        </section>
      )}

      {audioUrl && (
        <section className="panel">
          <h3>Interactive Waveform (WaveSurfer.js)</h3>
          <WavePlayer audioUrl={audioUrl} seekTo={seekTo} />
        </section>
      )}

      <section className="panel">
        <div className="row-between">
          <div>
            <h3>Unified Speaker & Sound Timeline</h3>
            <p className="muted" style={{ margin: 0 }}>Click any entry below to seek WaveSurfer playback directly to that timestamp.</p>
          </div>
          <div className="export-buttons">
            <Download size={14} color="#94a3b8" />
            {['srt', 'vtt', 'json', 'csv', 'txt'].map((f) => (
              <button key={f} onClick={() => exportResult(f, data).catch(() => alert('Export failed'))}>
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        {data.timeline.length === 0 ? (
          <p className="muted" style={{ marginTop: 12 }}>No speech or ambient sounds exceeded confidence 0.50. Expand filtered detections below to inspect raw predictions.</p>
        ) : (
          <div className="timeline">
            {data.timeline.map((item) => (
              <button key={item.id} className={`timeline-item ${item.category}`} onClick={() => onSeek(item.start)}>
                <span className="time">{item.start.toFixed(2)}s – {item.end.toFixed(2)}s</span>
                <b>{item.category === 'speech' ? (item.speaker || item.label || 'Speaker 1') : item.label}</b>
                <span>{item.display_text}</span>
                {item.confidence != null && <small>Confidence {(item.confidence * 100).toFixed(1)}% · {item.source}</small>}
              </button>
            ))}
          </div>
        )}
      </section>

      {dropped.length > 0 && (
        <section className="panel">
          <h3>Filtered / Suppressed Background Detections ({dropped.length})</h3>
          {dropped.slice(0, 12).map((d, i) => (
            <div key={i} className="list-card">
              <div>
                <b>{d.item?.label || 'Segment'}</b> {d.item?.confidence != null ? `(${(d.item.confidence * 100).toFixed(1)}%)` : ''}
                <div className="muted">{d.reason}</div>
              </div>
              <span className="time">{d.item?.start != null ? `${d.item.start.toFixed(2)}s` : ''}</span>
            </div>
          ))}
        </section>
      )}
    </div>
  )
}
