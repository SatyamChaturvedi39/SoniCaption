import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Bell,
  BookOpen,
  CheckCircle2,
  FileText,
  Mic,
  Plus,
  Radio,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  Square,
  Trash2,
  Volume2,
  VolumeX,
  X,
  Zap,
} from 'lucide-react'
import {
  createReminder,
  deleteReminder,
  deleteSession,
  getAlerts,
  getReminders,
  getSessions,
  getSimilarSessions,
  sendAgentTurn,
  sendWatchChunk,
  toggleReminder,
} from '../api'
import type { AlertItem, Analysis, ReminderItem, SessionLog } from '../types'
import { RAW_MIC_CONSTRAINTS, blobToWavFile, startContinuousPcmMonitor } from '../utils/wavEncoder'

export function AgentView({ onRemindersChanged }: { onRemindersChanged: () => void }) {
  const [turns, setTurns] = useState<{
    role: 'user' | 'agent'
    text: string
    sounds?: any[]
    pitch?: number
    matchedSessions?: { session_id: number; filename: string; summary: string }[]
  }[]>([
    {
      role: 'agent',
      text: 'Multimodal Voice Intelligence Copilot ready. Ask questions across your recorded lectures or calls, click quick prompts below to extract study guides or audit conversation dynamics, or say "Bookmark: ..." to log spoken concepts.',
    },
  ])
  const [text, setText] = useState('')
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const [speakReply, setSpeakReply] = useState(false)
  const [denoise, setDenoise] = useState(true)
  const [err, setErr] = useState('')
  const recRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  function speakText(msg: string) {
    if (!speakReply || !('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(msg)
    u.rate = 1.05
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
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data)
      }
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
            <h3>Voice Copilot &amp; Intelligence Auditor</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>
              Groq LLaMA-3 powered interactive audio Q&amp;A, automated study flashcards, and conversation forensics across all archive sessions.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <label className="toggle-chip">
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
              DSP Denoise
            </label>
            <button className="secondary-btn" onClick={() => setSpeakReply((v) => !v)}>
              {speakReply ? <Volume2 size={14} color="#06b6d4" /> : <VolumeX size={14} />}
              {speakReply ? 'TTS Active' : 'TTS Muted'}
            </button>
          </div>
        </div>

        {/* Quick Prompt Pills */}
        <div className="quick-prompts">
          {[
            { label: 'Summarize Audio Archive', icon: FileText, prompt: 'Provide a structured summary of all my recorded sessions, key themes, and duration stats.' },
            { label: 'Extract Study Flashcards & Hints', icon: BookOpen, prompt: 'Extract key academic concepts, exam hints, and formula highlights from my analyzed recordings into study flashcards.' },
            { label: 'Audit Call / Lecture Dynamics', icon: Zap, prompt: 'Audit the conversational dynamics: speaker turns, emphasis frequency, speech pace, and background noise quality.' },
            { label: 'Find Anomalies & Alerts', icon: AlertTriangle, prompt: 'Scan all recordings for urgent acoustic alerts, loud spikes, or unexpected sound events.' },
          ].map((item, idx) => (
            <button
              key={idx}
              className="quick-prompt-btn"
              disabled={busy}
              onClick={() => submitTurn(undefined, item.prompt)}
            >
              <item.icon size={13} color="#06b6d4" /> {item.label}
            </button>
          ))}
        </div>

        {err && <div className="notice-bar">{err}</div>}

        <div className="chat-box">
          {turns.map((t, i) => (
            <div key={i} className={`chat-msg ${t.role}`}>
              <div className="mono" style={{ fontSize: 10, color: '#94a3b8', marginBottom: 5 }}>
                <b style={{ color: t.role === 'user' ? '#38bdf8' : '#a855f7' }}>
                  {t.role === 'user' ? 'USER' : 'INTELLIGENCE COPILOT'}
                </b>
                {t.pitch ? ` · F0: ${t.pitch} Hz` : ''}
              </div>
              <div style={{ whiteSpace: 'pre-wrap' }}>{t.text}</div>
              {t.role === 'user' && t.sounds && t.sounds.length > 0 && (
                <div className="tag-row">
                  {t.sounds.map((s, idx) => (
                    <span key={idx} className="sound-chip">
                      [{s.label}] {Math.round((s.confidence || 0) * 100)}%
                    </span>
                  ))}
                </div>
              )}
              {t.role === 'agent' && t.matchedSessions && t.matchedSessions.length > 0 && (
                <div className="tag-row">
                  {t.matchedSessions.map((m) => (
                    <span key={m.session_id} className="tag ok mono">
                      Source: {m.filename}
                    </span>
                  ))}
                </div>
              )}
              {t.role === 'agent' && t.sounds && t.sounds.length > 0 && (
                <div className="tag-row">
                  {t.sounds.map((r, idx) => (
                    <span key={idx} className="tag purple">
                      <Sparkles size={11} style={{ marginRight: 4 }} />
                      Bookmarked: {r.title}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="input-bar">
          <button className={recording ? 'danger-btn' : 'primary-btn'} onClick={toggleMic} disabled={busy}>
            {recording ? <Square size={14} /> : <Mic size={14} />}
            {recording ? 'Stop & Send' : 'Speak'}
          </button>
          <input
            className="input-field"
            value={text}
            placeholder="Ask anything about your audio logs or say 'Bookmark: ...'"
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
  const [modelSource, setModelSource] = useState<'resnet' | 'hf-cnn'>('resnet')
  const [baselineRms, setBaselineRms] = useState(0)
  const [calibrating, setCalibrating] = useState(false)
  const [rmsDb, setRmsDb] = useState(-60)
  const [speechActive, setSpeechActive] = useState(false)
  const [urgent, setUrgent] = useState<AlertItem | null>(null)
  const [feed, setFeed] = useState<{ ts: string; msg: string; rms: number; isSpeech?: boolean; isEvent?: boolean }[]>([])
  const [err, setErr] = useState('')

  const stopCaptureRef = useRef<(() => void) | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const animFrameRef = useRef<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const denoiseRef = useRef(denoise)
  const modelRef = useRef(modelSource)
  const baselineRef = useRef(baselineRms)
  const calibratingRef = useRef(calibrating)
  const inFlightRef = useRef(false)

  useEffect(() => { denoiseRef.current = denoise }, [denoise])
  useEffect(() => { modelRef.current = modelSource }, [modelSource])
  useEffect(() => { baselineRef.current = baselineRms }, [baselineRms])
  useEffect(() => { calibratingRef.current = calibrating }, [calibrating])

  // Setup real-time oscilloscope
  const setupOscilloscope = (stream: MediaStream) => {
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
      audioCtxRef.current = audioCtx
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)

      const bufferLength = analyser.frequencyBinCount
      const dataArray = new Uint8Array(bufferLength)

      const draw = () => {
        animFrameRef.current = requestAnimationFrame(draw)
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        analyser.getByteTimeDomainData(dataArray)
        ctx.fillStyle = '#08080d'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        ctx.lineWidth = 2
        ctx.strokeStyle = speechActive ? '#10b981' : '#06b6d4'
        ctx.shadowBlur = 6
        ctx.shadowColor = speechActive ? 'rgba(16, 185, 129, 0.5)' : 'rgba(6, 182, 212, 0.5)'

        ctx.beginPath()
        const sliceWidth = (canvas.width * 1.0) / bufferLength
        let x = 0
        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0
          const y = (v * canvas.height) / 2
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
          x += sliceWidth
        }
        ctx.lineTo(canvas.width, canvas.height / 2)
        ctx.stroke()
        ctx.shadowBlur = 0
      }
      draw()
    } catch {
      // AudioContext fallback
    }
  }

  const stopOscilloscope = () => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => {})
      audioCtxRef.current = null
    }
  }

  async function handleChunk(wav: File) {
    if (inFlightRef.current) return
    inFlightRef.current = true
    try {
      const res = await sendWatchChunk(wav, denoiseRef.current, baselineRef.current, modelRef.current)
      setRmsDb(res.rms_db)
      if (calibratingRef.current && res.noise_rms != null) {
        setBaselineRms(res.noise_rms)
        setCalibrating(false)
        setSpeechActive(false)
        setFeed((prev) => [
          {
            ts: new Date().toLocaleTimeString(),
            msg: `Calibrated room floor (${res.rms_db.toFixed(1)} dB) — stationary fan noise suppressed`,
            rms: res.rms_db,
          },
          ...prev,
        ].slice(0, 20))
        return
      }
      setSpeechActive(Boolean(res.has_speech))
      if (res.urgent_alert) {
        setUrgent(res.urgent_alert)
        onAlertLogged()
      }
      const hasEvents = Boolean(res.events?.length)
      const tagText = hasEvents
        ? res.events.map((e: any) => `${e.label} (${Math.round(e.confidence * 100)}%)`).join(', ')
        : res.has_speech
        ? 'Speech detected (Silero VAD Active)'
        : 'Stationary ambient floor (Noise gate active)'
      setFeed((prev) => [
        {
          ts: new Date().toLocaleTimeString(),
          msg: tagText,
          rms: res.rms_db,
          isSpeech: Boolean(res.has_speech),
          isEvent: hasEvents,
        },
        ...prev,
      ].slice(0, 20))
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
      stopOscilloscope()
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
      setupOscilloscope(stream)
      stopCaptureRef.current = startContinuousPcmMonitor(stream, 2500, handleChunk)
    } catch {
      setErr('Microphone permission required for Live Monitor.')
    }
  }

  useEffect(() => () => {
    stopCaptureRef.current?.()
    stopOscilloscope()
    streamRef.current?.getTracks().forEach((t) => t.stop())
  }, [])

  const pct = Math.max(4, Math.min(100, ((rmsDb + 65) / 65) * 100))
  const floorDb = baselineRms > 0 ? `${(20 * Math.log10(baselineRms)).toFixed(1)} dB` : 'Auto (-33 dB)'

  return (
    <div className="content">
      {urgent && (
        <div className="urgent-bar">
          <ShieldAlert size={20} />
          <span>
            CRITICAL ALERT: {urgent.event_label} ({Math.round(urgent.confidence * 100)}%) — {urgent.snippet_text}
          </span>
          <button className="secondary-btn" style={{ marginLeft: 'auto' }} onClick={() => setUrgent(null)}>
            Acknowledge
          </button>
        </div>
      )}

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>Real-Time Ambient &amp; Speech Monitor</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>
              Continuous 2.5s streaming with real-time Web Audio oscilloscope, smoothed Spectral Subtraction, calibrated room floor, and dual CNN models.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <select
              className="input-field"
              style={{ width: 'auto', padding: '6px 12px', fontSize: 12 }}
              value={modelSource}
              onChange={(e) => setModelSource(e.target.value as 'resnet' | 'hf-cnn')}
            >
              <option value="resnet">Member 1: ResNet-18 (Local)</option>
              <option value="hf-cnn">Member 2: 4-Block CNN (HF)</option>
            </select>
            <label className="toggle-chip">
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
              Denoise Stream
            </label>
            {watching && (
              <button className="secondary-btn" onClick={() => setCalibrating(true)} disabled={calibrating}>
                {calibrating ? 'Sampling Room Noise...' : 'Calibrate Fan Floor'}
              </button>
            )}
            <button className={watching ? 'danger-btn' : 'primary-btn'} onClick={toggleWatch}>
              {watching ? <Square size={14} /> : <Radio size={14} />}
              {watching ? 'Stop Monitor' : 'Start Live Monitor'}
            </button>
          </div>
        </div>

        {err && <div className="notice-bar">{err}</div>}

        {/* Live Audio Telemetry & Oscilloscope */}
        <div>
          <div className="row-between" style={{ fontSize: 12 }}>
            <span>
              Live Level: <b className="mono" style={{ color: '#ffffff' }}>{rmsDb.toFixed(1)} dB</b> · Calibrated Floor:{' '}
              <b className="mono" style={{ color: '#06b6d4' }}>{floorDb}</b>
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className={`status-dot ${speechActive ? 'live' : ''}`} />
              VAD:{' '}
              <b style={{ color: speechActive ? '#10b981' : '#64748b' }}>
                {speechActive ? 'Speech Active' : 'Stationary Noise Gate'}
              </b>
            </span>
          </div>

          <div className="meter-bar">
            <div className="meter-fill" style={{ width: `${pct}%` }} />
          </div>

          <canvas ref={canvasRef} width={640} height={70} className="oscilloscope-canvas" />
        </div>

        <h4 style={{ margin: '18px 0 10px', fontSize: 13 }}>Live Event &amp; Acoustic Stream</h4>
        {feed.length === 0 ? (
          <p className="muted">
            Click &ldquo;Start Live Monitor&rdquo; to start streaming. In a classroom, click &ldquo;Calibrate Fan Floor&rdquo; to sample ambient noise, then test speaking, coughing, or clapping.
          </p>
        ) : (
          feed.map((f, i) => (
            <div
              key={i}
              className="list-card"
              style={{
                borderColor: f.isEvent ? 'rgba(245, 158, 11, 0.4)' : f.isSpeech ? 'rgba(16, 185, 129, 0.35)' : undefined,
                background: f.isEvent ? 'rgba(245, 158, 11, 0.05)' : f.isSpeech ? 'rgba(16, 185, 129, 0.04)' : undefined,
              }}
            >
              <div>
                <b style={{ color: f.isEvent ? '#fde68a' : f.isSpeech ? '#6ee7b7' : '#e2e8f0' }}>{f.msg}</b>
                <div className="muted mono">{f.ts}</div>
              </div>
              <span className="tag mono">{f.rms.toFixed(1)} dB</span>
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
  const [newDue, setNewDue] = useState('Key Concept')

  async function loadAll() {
    const [s, r, a] = await Promise.all([getSessions(search), getReminders(), getAlerts()])
    setSessions(s.sessions)
    setReminders(r.reminders)
    setAlerts(a.alerts)
  }

  useEffect(() => {
    loadAll().catch(() => {})
  }, [search, refreshKey])

  async function openWithMatches(s: SessionLog) {
    let similar = s.metrics?.similar_sessions || []
    try {
      const res = await getSimilarSessions(s.id)
      similar = res.matches
    } catch {}
    onOpenSession({
      session_id: s.id,
      filename: s.filename,
      duration: s.duration,
      sample_rate: 16000,
      mode: 'live',
      summary: s.summary,
      similar_sessions: similar,
      speech_segments: s.timeline.filter((t) => t.category === 'speech'),
      sound_events: s.timeline.filter((t) => t.category === 'sound_event'),
      timeline: s.timeline,
      metrics: s.metrics,
    })
  }

  return (
    <div className="content">
      <div className="grid-2">
        <div className="panel">
          <div className="panel-header">
            <h3>Audio Intelligence Archive ({sessions.length})</h3>
          </div>
          <div className="input-bar" style={{ marginBottom: 14, marginTop: 0 }}>
            <input
              className="input-field"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search transcript, topic, or sound..."
            />
            <span className="secondary-btn">
              <Search size={14} />
            </span>
          </div>
          {sessions.length === 0 ? (
            <p className="muted">No saved recordings yet. Analyze or record audio in the Studio to build your archive.</p>
          ) : (
            sessions.map((s) => (
              <div key={s.id} className="list-card" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                <div className="row-between">
                  <div>
                    <b style={{ color: '#ffffff' }}>{s.filename}</b> <span className="sound-chip">{s.source_type}</span>
                    <div className="muted mono">
                      {s.created_at} · {s.duration}s · SNR +{Math.max(0, s.snr_after_db - s.snr_before_db).toFixed(1)} dB
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button className="secondary-btn" onClick={() => openWithMatches(s)}>
                      Open
                    </button>
                    <button className="secondary-btn" onClick={() => deleteSession(s.id).then(loadAll)}>
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
                {s.summary && <div className="muted" style={{ marginTop: 6 }}>{s.summary}</div>}
              </div>
            ))
          )}
        </div>

        <div>
          <div className="panel">
            <h3 style={{ marginBottom: 10 }}>
              <Bell size={15} style={{ verticalAlign: 'middle', marginRight: 6, color: '#f59e0b' }} />
              Study Bookmarks &amp; Action Items ({reminders.filter((r) => r.status === 'pending').length} active)
            </h3>
            <div className="input-bar" style={{ marginBottom: 14, marginTop: 0 }}>
              <input
                className="input-field"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder="Add study note or action item..."
              />
              <input
                className="input-field"
                style={{ maxWidth: 120 }}
                value={newDue}
                onChange={(e) => setNewDue(e.target.value)}
                placeholder="Tag / Topic..."
              />
              <button
                className="primary-btn"
                onClick={async () => {
                  if (!newTitle.trim()) return
                  await createReminder({ title: newTitle.trim(), due_time: newDue })
                  setNewTitle('')
                  loadAll()
                }}
              >
                <Plus size={14} /> Save
              </button>
            </div>
            {reminders.length === 0 ? (
              <p className="muted">No study notes or voice bookmarks saved yet.</p>
            ) : (
              reminders.map((r) => (
                <div key={r.id} className={`reminder-row ${r.status === 'completed' ? 'done' : ''}`}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <input
                      type="checkbox"
                      checked={r.status === 'completed'}
                      onChange={() => toggleReminder(r.id, r.status === 'completed' ? 'pending' : 'completed').then(loadAll)}
                    />
                    <div>
                      <b style={{ color: '#ffffff' }}>{r.title}</b>
                      <div className="muted mono">
                        {r.due_time} · {r.priority} · {r.trigger_source}
                      </div>
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
            <h3 style={{ marginBottom: 10 }}>
              <AlertTriangle size={15} style={{ verticalAlign: 'middle', marginRight: 6, color: '#f43f5e' }} />
              Critical Acoustic Triggers ({alerts.length})
            </h3>
            {alerts.length === 0 ? (
              <p className="muted">No high-priority alerts triggered.</p>
            ) : (
              alerts.slice(0, 8).map((a) => (
                <div key={a.id} className="list-card">
                  <div>
                    <b style={{ color: '#fda4af' }}>{a.event_label}</b>{' '}
                    <span className="mono">({Math.round(a.confidence * 100)}%)</span>
                    <div className="muted">
                      {a.snippet_text} · {a.created_at}
                    </div>
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
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(5, 5, 10, 0.75)',
        backdropFilter: 'blur(16px)',
        display: 'grid',
        placeItems: 'center',
        zIndex: 50,
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        className="panel"
        style={{ maxWidth: 660, width: '100%', maxHeight: '88vh', overflowY: 'auto' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-header">
          <div>
            <h3>SonicPulse Multimodal Engine Architecture</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>
              MAI522-5 Speech Processing &amp; Recognition · Units 1–4 Integration
            </p>
          </div>
          <button className="secondary-btn" onClick={onClose}>
            <X size={14} /> Close
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 }}>
          <div className="list-card" style={{ flexDirection: 'column', alignItems: 'flex-start', margin: 0 }}>
            <span className="tag ok mono">Member 1 (Lead DSP &amp; Speech)</span>
            <div style={{ fontSize: 12, marginTop: 4, lineHeight: 1.45 }}>
              <b>Unit 1:</b> Pre-emphasis, Hamming/Hanning, Boll STFT Spectral Subtraction OLA, Levinson-Durbin LPC Formants.<br />
              <b>Unit 2:</b> 13-MFCC + $\Delta$ + $\Delta\Delta$, Autocorrelation F0 pitch contour, LBG-VQ speaker clustering.<br />
              <b>Unit 3:</b> Subsequence DTW Query-by-Voice keyword spotting.<br />
              <b>Unit 4:</b> Discrete HMM Viterbi discourse decoding.
            </div>
          </div>
          <div className="list-card" style={{ flexDirection: 'column', alignItems: 'flex-start', margin: 0 }}>
            <span className="tag warn mono">Member 2 (Deep Learning)</span>
            <div style={{ fontSize: 12, marginTop: 4, lineHeight: 1.45 }}>
              <b>ESC-50 Sound CNN:</b> 4-block PyTorch CNN + local ResNet-18.<br />
              <b>Dataset:</b> 10 environmental classes (clapping, laughter, sirens, footsteps, etc.).<br />
              <b>Validation:</b> Fold-4 Val: 88.75% · Fold-5 Test: 80.00% · Macro-F1: 0.787.<br />
              <b>Inference:</b> Hugging Face hub + local weights fallback.
            </div>
          </div>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>ESC-50 Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>Test Support</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r[0]}>
                <td>
                  <code>{r[0]}</code>
                </td>
                <td className="mono">{r[1]}</td>
                <td className="mono">{r[2]}</td>
                <td className="mono">
                  <b>{r[3]}</b>
                </td>
                <td className="mono">{r[4]}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{ marginTop: 14, fontSize: 12 }} className="tag ok">
          <CheckCircle2 size={13} /> SonicPulse Engine Pipeline Active &amp; Verified
        </div>
      </div>
    </div>
  )
}
