import { useEffect, useState } from 'react'
import type { Health } from './types'
import { getHealth } from './lib/api'
import LiveTab from './components/LiveTab'
import UploadTab from './components/UploadTab'

type Tab = 'live' | 'upload'

export default function App() {
  const [tab, setTab] = useState<Tab>('live')
  const [health, setHealth] = useState<Health | null>(null)
  const [down, setDown] = useState(false)
  useEffect(() => { getHealth().then(setHealth).catch(() => setDown(true)) }, [])
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><span className="logo" aria-hidden>◖◗</span><div><h1>SonicCaption</h1><p>Who spoke, what was said, and what else was in the room</p></div></div>
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'live'} className={tab === 'live' ? 'on' : ''} onClick={() => setTab('live')}><span className="rec-dot sm" /> Live recording</button>
          <button role="tab" aria-selected={tab === 'upload'} className={tab === 'upload' ? 'on' : ''} onClick={() => setTab('upload')}>Upload audio</button>
        </nav>
        <div className="health" title={health ? `Speaker model: ${health.speaker_encoder}` : ''}>
          {down ? <span className="pill st-alert">Backend offline</span> : health ? (
            <>
              <span className="pill st-ok">Server ready</span>
              <span className={`pill ${health.asr.available ? 'st-ok' : 'st-alert'}`}>{health.asr.available ? `Speech-to-text: ${health.asr.provider}` : 'Speech-to-text: not set up'}</span>
            </>
          ) : <span className="pill st-idle">Connecting…</span>}
        </div>
      </header>
      {health && !health.asr.available && (
        <div className="banner banner-red top-banner" role="alert">Speech-to-text is not configured. {health.asr.hint} Without it you still get speakers, sound effects and background noise, but no transcript, scenario or tasks.</div>
      )}
      <main>
        <div hidden={tab !== 'live'}><LiveTab /></div>
        <div hidden={tab !== 'upload'}><UploadTab /></div>
      </main>
      <footer>SonicCaption · runs on your machine · audio is only sent to Groq if you set GROQ_API_KEY</footer>
    </div>
  )
}
