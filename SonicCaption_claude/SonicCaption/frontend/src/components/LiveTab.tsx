import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Analysis, LiveEvent, Scenario, Task } from '../types'
import { startMic, MicHandle } from '../lib/liveAudio'
import { CATEGORY_NAME, mmss, speakerColor, STATUS_CLASS } from '../lib/fmt'
import Timeline, { Block } from './Timeline'
import Results from './Results'

type Phase = 'idle' | 'connecting' | 'recording' | 'analysing' | 'done'
interface Line { id: string; start: number; end: number; text: string; speaker: string; speaker_id: number }

export default function LiveTab() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [withVideo, setWithVideo] = useState(true)
  const [denoise, setDenoise] = useState(true)
  const [blocks, setBlocks] = useState<Record<string, Block>>({})
  const [lines, setLines] = useState<Line[]>([])
  const [scenario, setScenario] = useState<Scenario | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  const [notice, setNotice] = useState<{ tone: 'red' | 'yellow'; text: string } | null>(null)
  const [progress, setProgress] = useState<{ p: number; msg: string } | null>(null)
  const [result, setResult] = useState<Analysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const micRef = useRef<MicHandle | null>(null)
  const recRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const videoRef = useRef<HTMLVideoElement>(null)
  const t0 = useRef(0)

  useEffect(() => () => { micRef.current?.stop(); wsRef.current?.close() }, [])
  useEffect(() => {
    if (phase !== 'recording') return
    const id = setInterval(() => setElapsed((Date.now() - t0.current) / 1000), 250)
    return () => clearInterval(id)
  }, [phase])

  const onEvent = useCallback((e: LiveEvent) => {
    switch (e.type) {
      case 'ready':
        if (!e.asr.available) setNotice({ tone: 'red', text: `Speech-to-text is not configured, so there will be no transcript, scenario or tasks. ${e.asr.hint ?? ''} Noise, sound effects and speakers are still analysed live.` })
        break
      case 'status': if (e.asr === 'error') setNotice({ tone: 'yellow', text: e.message }); break
      case 'segment':
        setBlocks(prev => ({ ...prev, [e.id]: { id: e.id, start: e.start, end: e.end, category: e.category, label: e.label, status: e.status, speaker_id: e.speaker_id ?? null, speaker: e.speaker ?? null, confidence: e.confidence, open: e.open, text: prev[e.id]?.text } }))
        break
      case 'transcript':
        setLines(prev => [...prev.filter(l => l.id !== e.id), { id: e.id, start: e.start, end: e.end, text: e.text, speaker: e.speaker, speaker_id: e.speaker_id }].sort((a, b) => a.start - b.start))
        setBlocks(prev => (prev[e.id] ? { ...prev, [e.id]: { ...prev[e.id], text: e.text, speaker_id: e.speaker_id, speaker: e.speaker } } : prev))
        break
      case 'insights': setScenario(e.scenario); setTasks(e.tasks); break
      case 'progress': setProgress({ p: e.progress, msg: e.message }); break
      case 'final': setResult(e.result); setPhase('done'); setProgress(null); break
      case 'error': setError(e.message); setPhase(p => (p === 'done' ? p : 'idle')); break
    }
  }, [])

  const start = async () => {
    setError(null); setResult(null); setBlocks({}); setLines([]); setScenario(null); setTasks([]); setNotice(null); setVideoUrl(null); setElapsed(0)
    setPhase('connecting')
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws/live`)
      wsRef.current = ws
      ws.onmessage = m => onEvent(JSON.parse(m.data as string) as LiveEvent)
      ws.onerror = () => { setError('Could not reach the SonicCaption server. Is the backend running?'); setPhase('idle') }
      await new Promise<void>((res, rej) => { ws.onopen = () => res(); setTimeout(() => rej(new Error('Timed out connecting to the backend.')), 6000) })
      ws.send(JSON.stringify({ type: 'start', sample_rate: 16000, denoise, asr: true }))
      const mic = await startMic(withVideo, pcm => { if (ws.readyState === WebSocket.OPEN) ws.send(pcm.buffer) })
      micRef.current = mic
      if (videoRef.current && withVideo) { videoRef.current.srcObject = mic.stream; void videoRef.current.play().catch(() => undefined) }
      try {
        const rec = new MediaRecorder(mic.stream)
        chunksRef.current = []
        rec.ondataavailable = ev => { if (ev.data.size) chunksRef.current.push(ev.data) }
        rec.onstop = () => setVideoUrl(URL.createObjectURL(new Blob(chunksRef.current, { type: rec.mimeType })))
        rec.start(1000)
        recRef.current = rec
      } catch { /* recording a local copy is optional */ }
      t0.current = Date.now()
      setPhase('recording')
    } catch (err) {
      micRef.current?.stop(); wsRef.current?.close()
      setError(err instanceof Error ? (err.name === 'NotAllowedError' ? 'Camera/microphone permission was denied. Allow access in the browser address bar and try again.' : err.message) : String(err))
      setPhase('idle')
    }
  }

  const stop = () => {
    micRef.current?.stop(); micRef.current = null
    if (recRef.current && recRef.current.state !== 'inactive') recRef.current.stop()
    setPhase('analysing'); setProgress({ p: 0.02, msg: 'Finishing the live analysis…' })
    wsRef.current?.send(JSON.stringify({ type: 'stop' }))
  }

  const list = useMemo(() => Object.values(blocks).sort((a, b) => a.start - b.start), [blocks])
  const now = [...list].reverse().find(b => b.category !== 'silence' || b.open) ?? list[list.length - 1]
  const events = [...list].filter(b => b.category !== 'silence' && !(b.category === 'speech' && b.open)).reverse().slice(0, 14)
  const duration = Math.max(elapsed, list.length ? list[list.length - 1].end : 0, 10)

  return (
    <div className="tab-live">
      <div className="panel controls">
        <div className="control-left">
          {phase === 'idle' || phase === 'done'
            ? <button className="btn btn-primary btn-lg" onClick={() => void start()}><span className="rec-dot" /> {phase === 'done' ? 'Record again' : 'Start recording'}</button>
            : phase === 'recording'
              ? <button className="btn btn-stop btn-lg" onClick={stop}>■ Stop & analyse</button>
              : <button className="btn btn-lg" disabled>{phase === 'connecting' ? 'Connecting…' : 'Analysing…'}</button>}
          <label className="check"><input type="checkbox" checked={withVideo} disabled={phase !== 'idle' && phase !== 'done'} onChange={e => setWithVideo(e.target.checked)} /> Camera</label>
          <label className="check"><input type="checkbox" checked={denoise} disabled={phase !== 'idle' && phase !== 'done'} onChange={e => setDenoise(e.target.checked)} /> Reduce noise for speaker analysis</label>
        </div>
        <div className="clock" aria-live="polite">{phase === 'recording' && <span className="live-badge">LIVE</span>} {mmss(elapsed)}</div>
      </div>

      {error && <div className="banner banner-red" role="alert">{error}</div>}
      {notice && <div className={`banner banner-${notice.tone}`}>{notice.text}</div>}

      {(phase === 'recording' || phase === 'analysing' || (phase === 'done' && !result)) && (
        <div className="live-grid">
          <div className="panel video-panel">
            {withVideo ? <video ref={videoRef} muted playsInline className="video" /> : <div className="video audio-only">Audio only</div>}
            {now && phase === 'recording' && (
              <div className={`now-chip ${STATUS_CLASS[now.status]}`}>
                <small>Right now</small>
                <strong>{now.category === 'speech' ? `Speech${now.speaker ? ' · ' + now.speaker : ''}` : now.category === 'sound_effect' ? `Sound effect: ${now.label}` : now.category === 'background_noise' ? `Background noise: ${now.label}` : 'Silence'}</strong>
              </div>
            )}
          </div>
          <div className="panel feed">
            <h3>What was just detected</h3>
            {events.length === 0 && <p className="muted">Listening… the first result appears after about 1.5 seconds.</p>}
            <ul className="event-list">
              {events.map(b => (
                <li key={b.id} className={STATUS_CLASS[b.status]}>
                  <span className="when">{mmss(b.start)}–{mmss(b.end)}</span>
                  <span className="what"><strong>{b.category === 'speech' ? (b.speaker ?? 'Speech') : b.label}</strong> <em>{CATEGORY_NAME[b.category]}</em></span>
                  {b.text && <span className="said">“{b.text}”</span>}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {(phase === 'recording' || phase === 'analysing') && (
        <>
          <div className="panel"><Timeline blocks={list} duration={duration} follow={phase === 'recording'} /></div>
          <div className="live-grid">
            <div className="panel">
              <h3>Live transcript</h3>
              {lines.length === 0 && <p className="muted">Words appear here a few seconds after someone finishes a sentence.</p>}
              <div className="transcript">{lines.map(l => <div key={l.id} className="utt"><span className="utt-t">{mmss(l.start)}</span><span className="utt-who" style={{ color: speakerColor(l.speaker_id) }}>{l.speaker}</span><span className="utt-text">{l.text}</span></div>)}</div>
            </div>
            <div className="panel">
              <h3>Situation & tasks so far</h3>
              {scenario ? <p><span className="chip chip-yellow">{scenario.label}</span> <span className="muted">{Math.round(scenario.confidence * 100)}% sure</span></p> : <p className="muted">Waiting for enough conversation to tell what kind of situation this is.</p>}
              <ul className="mini-tasks">{tasks.map(t => <li key={t.id}><i className="dot" style={{ background: speakerColor(t.owner_id) }} /><strong>{t.owner}</strong> → {t.task}{t.due && <span className="chip chip-yellow">{t.due}</span>}</li>)}</ul>
            </div>
          </div>
        </>
      )}

      {phase === 'analysing' && progress && (
        <div className="panel"><div className="progress"><div style={{ width: `${Math.round(progress.p * 100)}%` }} /></div><p className="muted">{progress.msg} — a full re-analysis of the whole take gives the final, smoother result.</p></div>
      )}

      {result && (
        <>
          {videoUrl && <p><a className="btn btn-ghost" href={videoUrl} download="soniccaption-recording.webm">Download the recording</a></p>}
          <Results a={result} mediaUrl={videoUrl} isVideo={withVideo} />
        </>
      )}
      {phase === 'idle' && !result && !error && (
        <div className="panel hero"><h2>Record a conversation</h2><p>Press start and talk. Speech, sound effects (clap, rain, dog…) and background noise appear on the timeline as they happen, each person is tracked as a separate speaker, and tasks are assigned to whoever agreed to do them.</p></div>
      )}
    </div>
  )
}
