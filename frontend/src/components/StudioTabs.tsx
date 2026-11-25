import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Briefcase,
  CheckCircle2,
  ClipboardCheck,
  FileAudio,
  HelpCircle,
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
  deriveLiveMeetingIntelligence,
  sendAgentTurn,
  sendWatchChunk,
} from '../api'
import type { ActionItem, AlertItem, MeetingScenario } from '../types'
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
  const [liveQuestion, setLiveQuestion] = useState<string | null>(null)
  const [liveTurns, setLiveTurns] = useState<any[]>([])
  const [liveEvents, setLiveEvents] = useState<any[]>([])
  const [scenario, setScenario] = useState<MeetingScenario | null>(null)
  const [actionItems, setActionItems] = useState<ActionItem[]>([])
  const [feed, setFeed] = useState<{ ts: string; kind: 'speech' | 'sound' | 'sys'; speaker?: string; emphasis?: string; isQuestion?: boolean; qReason?: string; msg: string; cue?: string; rms: number }[]>([])
  const [err, setErr] = useState('')
  const stopCaptureRef = useRef<(() => void) | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const denoiseRef = useRef(denoise)
  const baselineRef = useRef(baselineRms)
  const calibratingRef = useRef(calibrating)
  const inFlightRef = useRef(false)
  const turnsRef = useRef<any[]>([])
  const eventsRef = useRef<any[]>([])
  const speakerProfilesRef = useRef<{ label: string; f0: number }[]>([])
  const [demoBusy, setDemoBusy] = useState(false)
  useEffect(() => { denoiseRef.current = denoise }, [denoise])
  useEffect(() => { baselineRef.current = baselineRms }, [baselineRms])
  useEffect(() => { calibratingRef.current = calibrating }, [calibrating])

  function assignLiveSpeaker(rawSpeaker?: string, meanF0?: number): string {
    if (rawSpeaker && rawSpeaker.trim()) return rawSpeaker
    const f0 = Number(meanF0 || 0)
    if (f0 < 45) return speakerProfilesRef.current[0]?.label || 'Speaker 1'
    const profiles = speakerProfilesRef.current
    if (profiles.length === 0) {
      profiles.push({ label: 'Speaker 1', f0 })
      return 'Speaker 1'
    }
    let bestIdx = 0
    let bestSemi = Infinity
    profiles.forEach((p, idx) => {
      const semi = Math.abs(12 * Math.log2(f0 / p.f0))
      if (semi < bestSemi) {
        bestSemi = semi
        bestIdx = idx
      }
    })
    if (bestSemi <= 2.8 || profiles.length >= 4) {
      profiles[bestIdx].f0 = profiles[bestIdx].f0 * 0.7 + f0 * 0.3
      return profiles[bestIdx].label
    }
    const label = `Speaker ${profiles.length + 1}`
    profiles.push({ label, f0 })
    return label
  }

  async function runCorporateDemo() {
    setDemoBusy(true)
    setErr('')
    try {
      speakerProfilesRef.current = []
      const nowStr = new Date().toLocaleTimeString()
      const demoTurns = [
        { speaker: 'Speaker 1', start: 0.5, mean_f0_hz: 132.0, emphasis_tag: 'Assertive', is_question: true, question_reason: 'Rising pitch contour (+22 Hz) + Question phrasing', text: 'Is the authentication pull request ready for staging deploy?' },
        { speaker: 'Speaker 2', start: 4.1, mean_f0_hz: 196.0, emphasis_tag: 'Neutral', is_question: false, text: 'Almost done — waiting on the final schema check.' },
        { speaker: 'Speaker 3', start: 8.2, mean_f0_hz: 158.0, emphasis_tag: 'Urgent', is_question: false, text: 'I will update GitHub tomorrow at 4 PM with the patch.' },
        { speaker: 'Speaker 1', start: 12.6, mean_f0_hz: 136.0, emphasis_tag: 'Emphatic', is_question: false, text: 'Awesome, Speaker 2 will share the QA checklist by Friday.' },
      ]
      const demoEvents = [
        { start: 7.5, end: 8.1, label: 'Laughter', confidence: 0.89, workplace_cue: 'Social cue: Room laughter / positive banter' },
        { start: 11.8, end: 12.4, label: 'Door knock', confidence: 0.91, workplace_cue: 'Entrance / desk knock cue — someone at door or seeking attention' },
      ]
      turnsRef.current = demoTurns
      eventsRef.current = demoEvents
      setLiveTurns(demoTurns)
      setLiveEvents(demoEvents)
      setLiveQuestion('Speaker 1: "Is the authentication pull request ready for staging deploy?" (Rising pitch contour (+22 Hz) + Question phrasing)')
      setFeed([
        ...demoEvents.map((e) => ({ ts: nowStr, kind: 'sound' as const, msg: `${e.label} (${Math.round(e.confidence * 100)}%)`, cue: e.workplace_cue, rms: -26.0 })),
        ...demoTurns.map((s) => ({ ts: nowStr, kind: 'speech' as const, speaker: s.speaker, emphasis: s.emphasis_tag, isQuestion: s.is_question, qReason: s.question_reason, msg: s.text, rms: -22.5 })),
      ].reverse())
      const intel = await deriveLiveMeetingIntelligence(demoTurns, demoEvents)
      if (intel.meeting_scenario) setScenario(intel.meeting_scenario)
      if (intel.action_items) setActionItems(intel.action_items)
    } catch (e: any) {
      setErr(e.message || 'Demo run failed')
    } finally {
      setDemoBusy(false)
    }
  }

  async function handleChunk(wav: File) {
    if (inFlightRef.current) return
    inFlightRef.current = true
    try {
      const res = await sendWatchChunk(wav, denoiseRef.current, baselineRef.current, 'resnet', true)
      setRmsDb(res.rms_db)
      if (calibratingRef.current && res.noise_rms != null) {
        setBaselineRms(res.noise_rms)
        setCalibrating(false)
        setSpeechActive(false)
        setFeed((prev) => [{ ts: new Date().toLocaleTimeString(), kind: 'sys' as const, msg: `Calibrated conference room floor (${res.rms_db.toFixed(1)} dB)`, rms: res.rms_db }, ...prev].slice(0, 20))
        return
      }
      setSpeechActive(Boolean(res.has_speech))
      if (res.urgent_alert) {
        setUrgent(res.urgent_alert)
        onAlertLogged()
      }

      const newItems: typeof feed = []
      const nowStr = new Date().toLocaleTimeString()

      const speechChunks: any[] = res.speech_chunks || []
      if (speechChunks.length > 0) {
        const mapped = speechChunks.map((s) => ({
          ...s,
          speaker: assignLiveSpeaker(s.speaker, s.mean_f0_hz),
        }))
        turnsRef.current = [...turnsRef.current, ...mapped].slice(-30)
        setLiveTurns(turnsRef.current)
        mapped.forEach((s) => {
          if (s.is_question) {
            setLiveQuestion(`${s.speaker}: "${s.text}" (${s.question_reason || 'Question'})`)
          }
          newItems.push({
            ts: nowStr,
            kind: 'speech',
            speaker: s.speaker,
            emphasis: s.emphasis_tag || 'Neutral',
            isQuestion: Boolean(s.is_question),
            qReason: s.question_reason,
            msg: s.text,
            rms: res.rms_db,
          })
        })
      }

      if (res.events?.length) {
        eventsRef.current = [...eventsRef.current, ...res.events].slice(-20)
        setLiveEvents(eventsRef.current)
        res.events.forEach((e: any) => {
          newItems.push({
            ts: nowStr,
            kind: 'sound',
            msg: `${e.label} (${Math.round(e.confidence * 100)}%)`,
            cue: e.workplace_cue,
            rms: res.rms_db,
          })
        })
      }

      if (newItems.length > 0) {
        setFeed((prev) => [...newItems.reverse(), ...prev].slice(0, 25))
        // Continuously refresh overarching meeting scenario & action commitments
        deriveLiveMeetingIntelligence(turnsRef.current, eventsRef.current)
          .then((intel) => {
            if (intel.meeting_scenario) setScenario(intel.meeting_scenario)
            if (intel.action_items) setActionItems(intel.action_items)
          })
          .catch(() => { })
      } else if (res.has_speech) {
        setFeed((prev) => [{ ts: nowStr, kind: 'sys' as const, msg: 'Voice activity detected — buffering phrase...', rms: res.rms_db }, ...prev].slice(0, 25))
      }
    } catch (e: any) {
      setErr(e.message || 'Live chunk failed')
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
      stopCaptureRef.current = startContinuousPcmMonitor(stream, 3000, handleChunk)
    } catch {
      setErr('Microphone permission required for Live Meeting Radar.')
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
          <span>ROOM ALERT: {urgent.event_label} ({Math.round(urgent.confidence * 100)}%) — {urgent.snippet_text}</span>
          <button className="secondary-btn" style={{ marginLeft: 'auto' }} onClick={() => setUrgent(null)}>Acknowledge</button>
        </div>
      )}

      {liveQuestion && (
        <div className="notice-bar" style={{ display: 'flex', alignItems: 'center', gap: 10, borderColor: 'rgba(6, 182, 212, 0.5)', background: 'rgba(6, 182, 212, 0.12)', color: '#cffafe' }}>
          <HelpCircle size={18} color="#22d3ee" />
          <span><b>QUESTION ASKED IN ROOM:</b> {liveQuestion}</span>
          <button className="secondary-btn" style={{ marginLeft: 'auto', padding: '4px 10px' }} onClick={() => setLiveQuestion(null)}>Dismiss</button>
        </div>
      )}

      <div className="panel" style={{ borderColor: 'rgba(6, 182, 212, 0.35)', background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(18, 21, 34, 0.95))' }}>
        <div className="row-between">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Briefcase size={20} color="#06b6d4" />
            <div>
              <small className="muted" style={{ textTransform: 'uppercase', letterSpacing: '0.07em', fontSize: 10 }}>Live Overarching Corporate Meeting Scenario</small>
              <h2 style={{ margin: 0, fontSize: 17 }}>{scenario ? scenario.scenario_label : 'Standby — Listening for Corporate Meeting Audio'}</h2>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <span className="tag ok">{scenario ? scenario.meeting_mood : 'Ready to Monitor'}</span>
            <span className="tag mono">{liveTurns.length} Remarks · {liveEvents.length} Room Cues</span>
          </div>
        </div>
        {scenario && scenario.active_cues?.length > 0 && (
          <div className="tag-row" style={{ marginTop: 10 }}>
            {scenario.active_cues.map((c, idx) => (
              <span key={idx} className="sound-chip">{c}</span>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>Live Corporate Meeting Radar — Workplace Accessibility HUD</h3>
            <p className="muted" style={{ margin: '3px 0 0' }}>
              Real-time speech transcription with one-word emphasis tags, vocal question inflection flags, action-item capture, and workplace room sound cues.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <label className="toggle-chip">
              <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />
              Denoise Stream
            </label>
            <button className="secondary-btn" onClick={runCorporateDemo} disabled={demoBusy}>
              {demoBusy ? 'Running Demo...' : 'Test Sample Meeting'}
            </button>
            {watching && (
              <button className="secondary-btn" onClick={() => setCalibrating(true)} disabled={calibrating}>
                {calibrating ? 'Sampling Room...' : 'Calibrate Room Floor'}
              </button>
            )}
            <button className={watching ? 'danger-btn' : 'primary-btn'} onClick={toggleWatch}>
              {watching ? <Square size={14} /> : <Mic size={14} />}
              {watching ? 'Stop Meeting Radar' : 'Start Meeting Radar'}
            </button>
          </div>
        </div>
        {err && <div className="notice-bar">{err}</div>}
        <div>
          <div className="row-between" style={{ fontSize: 12 }}>
            <span>Room Level: <b className="mono" style={{ color: '#06b6d4' }}>{rmsDb.toFixed(1)} dB</b> · Floor: <b className="mono">{floorDb}</b></span>
            <span>Speech Sensor: <b style={{ color: speechActive ? '#10b981' : '#64748b' }}>{speechActive ? 'Speaking Active' : 'Listening'}</b></span>
          </div>
          <div className="meter-bar"><div className="meter-fill" style={{ width: `${pct}%` }} /></div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3><ClipboardCheck size={16} style={{ verticalAlign: 'middle', marginRight: 7, color: '#10b981' }} />Live Spoken Action Items ({actionItems.length})</h3>
            <p className="muted" style={{ margin: '2px 0 0' }}>Automatically maps spoken commitments to Speaker, Deliverable, and Deadline in real time.</p>
          </div>
        </div>
        {actionItems.length === 0 ? (
          <p className="muted" style={{ margin: 0 }}>
            No commitments captured yet. Start the radar and say e.g. &ldquo;I will update GitHub tomorrow by 4 PM&rdquo;, or click &ldquo;Test Sample Meeting&rdquo;.
          </p>
        ) : (
          <table className="data-table">
            <thead><tr><th>Owner</th><th>Assigned Deliverable</th><th>Deadline</th><th>Priority</th></tr></thead>
            <tbody>
              {actionItems.map((a, idx) => (
                <tr key={idx}>
                  <td><span className="tag ok">{a.owner}</span></td>
                  <td><b>{a.task}</b></td>
                  <td><span className="sound-chip">{a.deadline}</span></td>
                  <td className="mono">{a.priority}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <div className="panel-header"><h3>Real-Time Workplace Transcript &amp; Room Cues</h3></div>
        {feed.length === 0 ? (
          <p className="muted">Click &ldquo;Start Meeting Radar&rdquo; (and &ldquo;Calibrate Room Floor&rdquo; in a loud office), then capture live conversation, questions, commitments, laughter, or door knocks.</p>
        ) : (
          feed.map((f, i) => (
            <div key={i} className="list-card" style={{ borderLeft: f.kind === 'speech' ? '3px solid #06b6d4' : f.kind === 'sound' ? '3px solid #f59e0b' : '1px solid var(--border)' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap' }}>
                  {f.speaker && <span className="tag ok">{f.speaker}</span>}
                  {f.emphasis && <span className="tag" style={{ borderColor: 'rgba(168,85,247,0.4)', color: '#d8b4fe' }}>{f.emphasis}</span>}
                  {f.isQuestion && <span className="sound-chip">? QUESTION · {f.qReason}</span>}
                  <b>{f.msg}</b>
                </div>
                {f.cue && <div className="muted" style={{ marginTop: 3 }}>{f.cue}</div>}
                <div className="muted mono" style={{ marginTop: 2 }}>{f.ts}</div>
              </div>
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
