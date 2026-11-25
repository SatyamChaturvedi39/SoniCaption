import { useEffect, useRef, useState } from 'react'
import {
  Bell,
  Bot,
  CheckCircle2,
  Mic,
  Plus,
  Search,
  Send,
  ShieldAlert,
  Square,
  Trash2,
  Upload,
  Volume2,
  VolumeX,
} from 'lucide-react'
import {
  createReminder,
  deleteReminder,
  deleteSession,
  getAlerts,
  getDtwLab,
  getHmmLab,
  getReminders,
  getSessions,
  getSignalLab,
  sendAgentTurn,
  sendWatchChunk,
  toggleReminder,
} from '../api'
import type { AlertItem, Analysis, ReminderItem, SessionLog } from '../types'
import { blobToWavFile } from '../utils/wavEncoder'
import { DualLineCanvas, DtwCanvas, HmmTrellisSvg, SpectrogramCanvas } from './VisualCanvas'

export function AgentView({ onRemindersChanged }: { onRemindersChanged: () => void }) {
  const [turns, setTurns] = useState<{ role: 'user' | 'agent'; text: string; sounds?: any[]; pitch?: number }[]>([
    {
      role: 'agent',
      text: 'Hi! I am SonicCaption Copilot. Unlike regular voice bots, I listen to both your words and background acoustics (door knocks, sirens, laughter, coughing). Speak or type to create smart reminders or ask about active logs.',
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
          text: res.transcript || userInput || '(ambient audio captured)',
          sounds: res.sound_events,
          pitch: res.pitch_stats?.mean_f0_hz,
        },
        { role: 'agent', text: res.reply, sounds: res.new_reminders },
      ])
      speakText(res.reply)
      onRemindersChanged()
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
        const wav = await blobToWavFile(blob, 'copilot-voice')
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
        <div className="row-between" style={{ marginBottom: 14 }}>
          <div>
            <h3><Bot size={18} style={{ verticalAlign: 'middle', marginRight: 8 }} />Acoustic-Aware Voice AI Copilot</h3>
            <p className="muted" style={{ margin: 0 }}>Speaks back, hears ambient context alongside speech, and logs reminders directly to SQLite.</p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <label className="toggle-chip">
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
              DSP Denoise
            </label>
            <button className="secondary-btn" onClick={() => setSpeakReply((v) => !v)}>
              {speakReply ? <Volume2 size={15} /> : <VolumeX size={15} />}
              {speakReply ? 'TTS On' : 'TTS Muted'}
            </button>
          </div>
        </div>

        {err && <div className="banner-warn">{err}</div>}

        <div className="chat-box">
          {turns.map((t, i) => (
            <div key={i} className={`chat-msg ${t.role}`}>
              <div style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>
                {t.role === 'user' ? 'YOU' : 'SONICCAPTION COPILOT'}
                {t.pitch ? ` · Mean F0 ${t.pitch} Hz` : ''}
              </div>
              <div>{t.text}</div>
              {t.role === 'user' && t.sounds && t.sounds.length > 0 && (
                <div className="tag-row">
                  {t.sounds.map((s, idx) => (
                    <span key={idx} className="sound-chip">[{s.label}] {Math.round((s.confidence || 0) * 100)}%</span>
                  ))}
                </div>
              )}
              {t.role === 'agent' && t.sounds && t.sounds.length > 0 && (
                <div className="tag-row">
                  {t.sounds.map((r, idx) => (
                    <span key={idx} className="sound-chip" style={{ background: 'rgba(16,185,129,0.16)', color: '#34d399', borderColor: 'rgba(16,185,129,0.35)' }}>
                      + Saved Reminder: {r.title}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="input-bar">
          <button className={recording ? 'danger-btn' : 'primary-btn'} onClick={toggleMic} disabled={busy}>
            {recording ? <Square size={15} /> : <Mic size={15} />}
            {recording ? 'Stop & Send' : 'Speak Turn'}
          </button>
          <input
            className="input-field"
            value={text}
            placeholder="Or type e.g. 'Remind me to review lab report tomorrow at 9 am'..."
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
            <Send size={15} /> Send
          </button>
        </div>
      </div>
    </div>
  )
}

export function WatchView({ onAlertLogged }: { onAlertLogged: () => void }) {
  const [watching, setWatching] = useState(false)
  const [denoise, setDenoise] = useState(true)
  const [rmsDb, setRmsDb] = useState(-60)
  const [speechActive, setSpeechActive] = useState(false)
  const [urgent, setUrgent] = useState<AlertItem | null>(null)
  const [feed, setFeed] = useState<{ ts: string; msg: string; rms: number }[]>([])
  const [err, setErr] = useState('')
  const timerRef = useRef<number | undefined>(undefined)
  const streamRef = useRef<MediaStream | null>(null)

  async function recordAndSendChunk(stream: MediaStream) {
    const rec = new MediaRecorder(stream)
    const chunks: BlobPart[] = []
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }
    rec.onstop = async () => {
      const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' })
      if (!blob.size) return
      try {
        const wav = await blobToWavFile(blob, 'watch')
        const res = await sendWatchChunk(wav, denoise)
        setRmsDb(res.rms_db)
        setSpeechActive(Boolean(res.has_speech))
        if (res.urgent_alert) {
          setUrgent(res.urgent_alert)
          onAlertLogged()
        }
        const tagText = res.events?.length
          ? res.events.map((e: any) => `${e.label} (${Math.round(e.confidence * 100)}%)`).join(', ')
          : res.has_speech ? 'Active human speech' : 'Ambient room baseline'
        setFeed((prev) => [{ ts: new Date().toLocaleTimeString(), msg: tagText, rms: res.rms_db }, ...prev].slice(0, 15))
      } catch (e: any) {
        setErr(e.message || 'Watch chunk failed')
      }
    }
    rec.start()
    window.setTimeout(() => { if (rec.state === 'recording') rec.stop() }, 2200)
  }

  async function toggleWatch() {
    if (watching) {
      if (timerRef.current) clearInterval(timerRef.current)
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setWatching(false)
      return
    }
    setErr('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      setWatching(true)
      recordAndSendChunk(stream)
      timerRef.current = window.setInterval(() => recordAndSendChunk(stream), 2700)
    } catch {
      setErr('Microphone permission required for Live Watch Mode.')
    }
  }

  useEffect(() => () => {
    if (timerRef.current) clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
  }, [])

  const pct = Math.max(4, Math.min(100, ((rmsDb + 65) / 65) * 100))

  return (
    <div className="content">
      {urgent && (
        <div className="banner-urgent">
          <ShieldAlert size={20} />
          <span>URGENT ACOUSTIC ALERT: {urgent.event_label} ({Math.round(urgent.confidence * 100)}%) — {urgent.snippet_text}</span>
          <button className="secondary-btn" style={{ marginLeft: 'auto' }} onClick={() => setUrgent(null)}>Acknowledge</button>
        </div>
      )}
      <div className="panel">
        <div className="row-between">
          <div>
            <h3>Live Ambient Safety & Meeting Monitor</h3>
            <p className="muted" style={{ margin: 0 }}>Streams rolling 2.2s microphone buffers through Spectral Subtraction + Silero VAD + 10-Class ResNet-18.</p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <label className="toggle-chip">
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
              Denoise Stream
            </label>
            <button className={watching ? 'danger-btn' : 'primary-btn'} onClick={toggleWatch}>
              {watching ? <Square size={15} /> : <Mic size={15} />}
              {watching ? 'Stop Watch Mode' : 'Start Live Monitor'}
            </button>
          </div>
        </div>
        {err && <div className="banner-warn" style={{ marginTop: 12 }}>{err}</div>}
        <div style={{ marginTop: 18 }}>
          <div className="row-between" style={{ fontSize: 12 }}>
            <span>Input Loudness: <b>{rmsDb.toFixed(1)} dB</b></span>
            <span>VAD Status: <b style={{ color: speechActive ? '#34d399' : '#94a3b8' }}>{speechActive ? 'Speech Active' : 'No Speech'}</b></span>
          </div>
          <div className="meter-bar"><div className="meter-fill" style={{ width: `${pct}%` }} /></div>
        </div>
        <h4 style={{ marginTop: 18, marginBottom: 8 }}>Live Acoustic Event Stream</h4>
        {feed.length === 0 ? (
          <p className="muted">Press Start Live Monitor to test background knocks, clapping, sirens, laughing, or speech in real time.</p>
        ) : (
          feed.map((f, i) => (
            <div key={i} className="list-card">
              <div><b>{f.msg}</b><div className="muted">{f.ts}</div></div>
              <span className="pill">{f.rms} dB</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export function HistoryAndRemindersView({
  onOpenSession,
  refreshKey,
}: {
  onOpenSession: (sessionAsAnalysis: Analysis) => void
  refreshKey: number
}) {
  const [sessions, setSessions] = useState<SessionLog[]>([])
  const [reminders, setReminders] = useState<ReminderItem[]>([])
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [search, setSearch] = useState('')
  const [newTitle, setNewTitle] = useState('')
  const [newDue, setNewDue] = useState('Today')

  async function loadAll() {
    const [s, r, a] = await Promise.all([getSessions(search), getReminders(), getAlerts()])
    setSessions(s.sessions)
    setReminders(r.reminders)
    setAlerts(a.alerts)
  }

  useEffect(() => { loadAll().catch(() => {}) }, [search, refreshKey])

  return (
    <div className="content">
      <div className="grid-2">
        <div className="panel">
          <div className="row-between" style={{ marginBottom: 10 }}>
            <h3>Call & Session Logs ({sessions.length})</h3>
          </div>
          <div className="input-bar" style={{ marginBottom: 14, marginTop: 0 }}>
            <input
              className="input-field"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search transcript, filename, or event..."
            />
            <span className="secondary-btn"><Search size={14} /></span>
          </div>
          {sessions.length === 0 ? (
            <p className="muted">No saved audio logs yet. Analyze an upload, live recording, or voice copilot turn.</p>
          ) : (
            sessions.map((s) => (
              <div key={s.id} className="list-card" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                <div className="row-between">
                  <div>
                    <b>{s.filename}</b> <span className="sound-chip">{s.source_type}</span>
                    <div className="muted">{s.created_at} · {s.duration}s · SNR +{Math.max(0, s.snr_after_db - s.snr_before_db).toFixed(1)} dB</div>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="secondary-btn"
                      onClick={() =>
                        onOpenSession({
                          session_id: s.id,
                          filename: s.filename,
                          duration: s.duration,
                          sample_rate: 16000,
                          mode: 'live',
                          summary: s.summary,
                          speech_segments: s.timeline.filter((t) => t.category === 'speech'),
                          sound_events: s.timeline.filter((t) => t.category === 'sound_event'),
                          timeline: s.timeline,
                          metrics: s.metrics,
                        })
                      }
                    >
                      Inspect
                    </button>
                    <button className="secondary-btn" onClick={() => deleteSession(s.id).then(loadAll)}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
                {s.summary && <div className="muted" style={{ color: '#cbd5e1' }}>{s.summary}</div>}
              </div>
            ))
          )}
        </div>

        <div>
          <div className="panel">
            <h3><Bell size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />Smart Reminders Queue ({reminders.filter((r) => r.status === 'pending').length} pending)</h3>
            <div className="input-bar" style={{ marginBottom: 14 }}>
              <input className="input-field" value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="Add manual reminder..." />
              <input className="input-field" style={{ maxWidth: 110 }} value={newDue} onChange={(e) => setNewDue(e.target.value)} placeholder="Due..." />
              <button
                className="primary-btn"
                onClick={async () => {
                  if (!newTitle.trim()) return
                  await createReminder({ title: newTitle.trim(), due_time: newDue })
                  setNewTitle('')
                  loadAll()
                }}
              >
                <Plus size={14} /> Add
              </button>
            </div>
            {reminders.length === 0 ? (
              <p className="muted">No reminders yet. Say "Remind me to..." in Studio or Voice Copilot to auto-extract tasks.</p>
            ) : (
              reminders.map((r) => (
                <div key={r.id} className={`reminder-item ${r.status === 'completed' ? 'done' : ''}`}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <input
                      type="checkbox"
                      checked={r.status === 'completed'}
                      onChange={() => toggleReminder(r.id, r.status === 'completed' ? 'pending' : 'completed').then(loadAll)}
                    />
                    <div>
                      <b>{r.title}</b>
                      <div className="muted">Due: {r.due_time} · {r.priority} · {r.trigger_source}</div>
                    </div>
                  </div>
                  <button className="secondary-btn" onClick={() => deleteReminder(r.id).then(loadAll)}>
                    <Trash2 size={13} />
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="panel">
            <h3>Ambient Security Alerts ({alerts.length})</h3>
            {alerts.length === 0 ? (
              <p className="muted">No urgent sirens, knocks, or breaking glass alerts logged yet.</p>
            ) : (
              alerts.slice(0, 8).map((a) => (
                <div key={a.id} className="list-card">
                  <div>
                    <b>{a.event_label}</b> ({Math.round(a.confidence * 100)}%)
                    <div className="muted">{a.snippet_text} · {a.created_at}</div>
                  </div>
                  <span className="sound-chip">{a.severity}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export function LabsAndModelView({ kind }: { kind: 'signal' | 'pattern' | 'hmm' | 'model' }) {
  const [data, setData] = useState<any>(null)
  const [preset, setPreset] = useState('weather')
  const [obsInput, setObsInput] = useState('')
  const [sigFile, setSigFile] = useState<File | null>(null)
  const [dtwA, setDtwA] = useState<File | null>(null)
  const [dtwB, setDtwB] = useState<File | null>(null)

  async function refreshLab(p = preset) {
    if (kind === 'signal') setData(await getSignalLab(sigFile, true))
    else if (kind === 'pattern') setData(await getDtwLab(dtwA, dtwB))
    else if (kind === 'hmm') {
      const obs = obsInput.trim() ? obsInput.split(',').map((s) => s.trim()).filter(Boolean) : undefined
      setData(await getHmmLab(p, obs))
    }
  }

  useEffect(() => {
    if (kind !== 'model') refreshLab().catch(() => {})
  }, [kind])

  if (kind === 'model') {
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
      <div className="content">
        <div className="panel">
          <div className="row-between">
            <div>
              <h3>SonicAcousticResNet-18 — Verified 10-Class Model Card</h3>
              <p className="muted" style={{ margin: 0 }}>Trained leak-free on ESC-50 (Folds 1–3 Train · Fold 4 Val · Strictly Unseen Fold 5 Test).</p>
            </div>
            <span className="pill live"><CheckCircle2 size={14} /> Active Weights Loaded</span>
          </div>
          <div className="metric-grid">
            <div className="metric"><small>Fold-4 Val Peak</small><b>88.75%</b></div>
            <div className="metric"><small>Fold-5 Held-Out Test</small><b>80.00%</b></div>
            <div className="metric"><small>Macro-Avg F1 (Fold 5)</small><b>0.787</b></div>
            <div className="metric"><small>Parameters / Input</small><b>11.24M · 128-Mel</b></div>
          </div>
          <table className="data-table">
            <thead><tr><th>ESC-50 Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Fold-5 Support</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r[0]}><td><code>{r[0]}</code></td><td>{r[1]}</td><td>{r[2]}</td><td><b>{r[3]}</b></td><td>{r[4]}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  if (kind === 'signal') {
    return (
      <div className="content">
        <div className="panel">
          <div className="row-between">
            <div>
              <h3>{data?.label || 'Signal & Spectral Enhancement Lab (Units 1–2)'}</h3>
              <p className="muted" style={{ margin: 0 }}>Inspects your last analyzed studio audio or any custom clip with Member 1 spectral subtraction, STFT, MFCC, LPC formants, and F0 contour.</p>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <label className="secondary-btn">
                <Upload size={14} /> {sigFile ? sigFile.name : 'Test Custom WAV'}
                <input type="file" accept=".wav,audio/*" hidden onChange={(e) => setSigFile(e.target.files?.[0] || null)} />
              </label>
              <button className="primary-btn" onClick={() => refreshLab()}>Run DSP Lab</button>
            </div>
          </div>
          <div className="metric-grid">
            <div className="metric"><small>SNR Before Denoise</small><b>{data?.snr_before_db ?? '—'} dB</b></div>
            <div className="metric"><small>SNR After Denoise</small><b>{data?.snr_after_db ?? '—'} dB</b></div>
            <div className="metric"><small>Active Pitch (F0)</small><b>{data?.f0_mean_hz ?? '—'} Hz</b></div>
            <div className="metric"><small>LPC Formants (Hz)</small><b>{data?.lpc_formants_hz?.slice(0, 3).join(', ') || '—'}</b></div>
          </div>
          <div className="grid-2">
            <DualLineCanvas primary={data?.waveform || []} secondary={data?.waveform_denoised || []} labelA="Raw Audio" labelB="Denoised (Boll OLA)" />
            <SpectrogramCanvas matrix={data?.stft_matrix || []} title="2D STFT Power Spectrogram (dB)" />
            <SpectrogramCanvas matrix={data?.mfcc || []} title="13-Band MFCC Cepstral Matrix" />
            <DualLineCanvas primary={data?.fft_mag_db || []} secondary={data?.lpc_envelope_db || []} labelA="FFT Spectrum (dB)" labelB="LPC Vocal-Tract Envelope (dB)" />
          </div>
        </div>
      </div>
    )
  }

  if (kind === 'pattern') {
    return (
      <div className="content">
        <div className="panel">
          <div className="row-between">
            <div>
              <h3>{data?.label || 'Dynamic Time Warping Pattern Studio (Unit 3)'}</h3>
              <p className="muted" style={{ margin: 0 }}>Pairwise MFCC cosine/Euclidean frame alignment with dynamic programming backtracking.</p>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <label className="secondary-btn">Clip A: {dtwA ? dtwA.name : 'Synthetic'}<input type="file" accept=".wav" hidden onChange={(e) => setDtwA(e.target.files?.[0] || null)} /></label>
              <label className="secondary-btn">Clip B: {dtwB ? dtwB.name : 'Synthetic'}<input type="file" accept=".wav" hidden onChange={(e) => setDtwB(e.target.files?.[0] || null)} /></label>
              <button className="primary-btn" onClick={() => refreshLab()}>Compute DTW</button>
            </div>
          </div>
          <div className="metric-grid">
            <div className="metric"><small>Total DTW Cost</small><b>{data?.distance ?? '—'}</b></div>
            <div className="metric"><small>Normalized Distance</small><b>{data?.normalized_distance ?? '—'}</b></div>
            <div className="metric"><small>Warp Path Steps</small><b>{data?.path_length ?? '—'}</b></div>
            <div className="metric"><small>Frame Grid</small><b>{data?.frames_a ? `${data.frames_a} × ${data.frames_b}` : '—'}</b></div>
          </div>
          <DtwCanvas matrix={data?.cost_matrix_sample || []} path={data?.path || []} />
        </div>
      </div>
    )
  }

  return (
    <div className="content">
      <div className="panel">
        <div className="row-between">
          <div>
            <h3>Interactive HMM & Viterbi Trellis Lab (Unit 4)</h3>
            <p className="muted" style={{ margin: 0 }}>Powered by Member 2's log-space Forward likelihood & Viterbi backpointer decoder (`algorithms/hmm.py`).</p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {['weather', 'coin_toss', 'urn_ball'].map((p) => (
              <button
                key={p}
                className={preset === p ? 'primary-btn' : 'secondary-btn'}
                onClick={() => { setPreset(p); setObsInput(''); refreshLab(p) }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
        <div className="input-bar">
          <input
            className="input-field"
            value={obsInput}
            onChange={(e) => setObsInput(e.target.value)}
            placeholder={`Custom observations (e.g. ${data?.model?.symbols?.join(', ') || 'walk, shop, clean'})`}
          />
          <button className="primary-btn" onClick={() => refreshLab()}>Decode Sequence</button>
        </div>
        <div className="metric-grid">
          <div className="metric"><small>Optimal States</small><b>{data?.best_path?.join(' → ') || '—'}</b></div>
          <div className="metric"><small>Forward Log-Likelihood</small><b>{data?.log_likelihood != null ? data.log_likelihood.toFixed(3) : '—'}</b></div>
          <div className="metric"><small>Viterbi Log-Prob</small><b>{data?.best_path_log_prob != null ? data.best_path_log_prob.toFixed(3) : '—'}</b></div>
          <div className="metric"><small>States Count</small><b>{data?.model?.states?.length ?? '—'}</b></div>
        </div>
        <HmmTrellisSvg data={data} />
      </div>
    </div>
  )
}
