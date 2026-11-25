import { useEffect, useRef, useState } from 'react'
import type { Analysis } from '../types'
import { analyzeFile, listSessions, loadSession } from '../lib/api'
import Results from './Results'

type Recent = Awaited<ReturnType<typeof listSessions>>['sessions']

export default function UploadTab() {
  const [file, setFile] = useState<File | null>(null)
  const [denoise, setDenoise] = useState(true)
  const [speakers, setSpeakers] = useState('auto')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState({ p: 0, msg: '' })
  const [result, setResult] = useState<Analysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [over, setOver] = useState(false)
  const [recent, setRecent] = useState<Recent>([])
  const [mediaUrl, setMediaUrl] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const refresh = () => { void listSessions().then(r => setRecent(r.sessions)).catch(() => undefined) }
  useEffect(refresh, [])

  const run = async () => {
    if (!file) return
    setBusy(true); setError(null); setResult(null); setMediaUrl(URL.createObjectURL(file)); setProgress({ p: 0.01, msg: 'Uploading…' })
    try {
      setResult(await analyzeFile(file, { denoise, speakerMode: speakers }, (p, msg) => setProgress({ p, msg })))
      refresh()
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(false) }
  }

  return (
    <div className="tab-upload">
      <div className="panel">
        <div
          className={`drop ${over ? 'over' : ''}`}
          onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
          onDrop={e => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) setFile(f) }}
          onClick={() => input.current?.click()} role="button" tabIndex={0} onKeyDown={e => { if (e.key === 'Enter') input.current?.click() }}
        >
          <input ref={input} type="file" hidden accept="audio/*,video/*,.wav,.mp3,.m4a,.ogg,.flac,.webm,.mp4" onChange={e => setFile(e.target.files?.[0] ?? null)} />
          {file ? <><strong>{file.name}</strong><span className="muted">{(file.size / 1e6).toFixed(1)} MB · click to choose another</span></> : <><strong>Drop an audio or video file here</strong><span className="muted">or click to browse · wav, mp3, m4a, ogg, flac, webm, mp4</span></>}
        </div>
        <div className="options">
          <label className="check"><input type="checkbox" checked={denoise} onChange={e => setDenoise(e.target.checked)} /> Reduce noise for speaker analysis</label>
          <label className="check">Speakers
            <select value={speakers} onChange={e => setSpeakers(e.target.value)}>
              <option value="auto">Detect automatically</option><option value="1">Just one</option><option value="2">Two</option><option value="3">Three</option><option value="4">Four</option><option value="5">Five</option>
            </select>
          </label>
          <button className="btn btn-primary" disabled={!file || busy} onClick={() => void run()}>{busy ? 'Analysing…' : 'Analyse'}</button>
        </div>
        {busy && <div className="progress-wrap"><div className="progress"><div style={{ width: `${Math.round(progress.p * 100)}%` }} /></div><p className="muted">{progress.msg}</p></div>}
        {error && <div className="banner banner-red" role="alert">{error}</div>}
      </div>

      {result && <Results a={result} mediaUrl={mediaUrl} isVideo={!!file && file.type.startsWith('video')} />}

      {!result && recent.length > 0 && (
        <div className="panel">
          <h3>Recent analyses</h3>
          <ul className="recent">{recent.map(s => (
            <li key={s.id}><button className="link" onClick={() => void loadSession(s.id).then(r => { setMediaUrl(null); setResult(r) }).catch(e => setError(String(e)))}>{s.filename}</button><span className="muted">{s.created_at} · {s.source_type} · {s.duration.toFixed(0)}s · {s.summary}</span></li>
          ))}</ul>
        </div>
      )}
    </div>
  )
}
