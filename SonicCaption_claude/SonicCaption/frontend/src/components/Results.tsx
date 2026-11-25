import { useEffect, useMemo, useRef, useState } from 'react'
import type { Analysis, Task } from '../types'
import { Bars, Heatmap, LineChart } from './Charts'
import Timeline, { Block } from './Timeline'
import { CATEGORY_NAME, mmss, pct, speakerColor } from '../lib/fmt'
import { exportTimeline, listTasks, setTaskStatus, StoredTask } from '../lib/api'

const TABS = ['Speakers', 'Tasks', 'Sounds & noise', 'MFCC & spectrogram', 'HMM', 'Voice patterns'] as const
type Tab = (typeof TABS)[number]

export function toBlocks(a: Analysis): Block[] {
  return a.timeline.map(t => ({ id: t.id, start: t.start, end: t.end, category: t.category, label: t.label, status: t.status, speaker_id: t.speaker_id, speaker: t.speaker, text: t.text, confidence: t.confidence }))
}

export default function Results({ a, mediaUrl, isVideo }: { a: Analysis; mediaUrl?: string | null; isVideo?: boolean }) {
  const [tab, setTab] = useState<Tab>('Speakers')
  const [sel, setSel] = useState<Block | null>(null)
  const media = useRef<HTMLMediaElement | null>(null)
  const seek = (t: number) => { const m = media.current; if (m) { m.currentTime = Math.max(0, t); void m.play().catch(() => undefined) } }
  const select = (b: Block) => { setSel(b); seek(b.start) }
  const blocks = useMemo(() => toBlocks(a), [a])
  const sc = a.nlu.scenario
  return (
    <section className="results">
      {a.asr.status !== 'ok' && (
        <div className="banner banner-red" role="alert">
          <strong>{a.asr.status === 'unavailable' ? 'No transcript: speech-to-text did not run.' : 'No speech to transcribe.'}</strong>{' '}
          {a.asr.status === 'skipped' ? a.asr.warnings.join(' ') : a.asr.error && <code>{a.asr.error}</code>}
          {a.asr.status === 'unavailable' && <div>Add a <code>GROQ_API_KEY</code> to <code>.env</code> (or install <code>faster-whisper</code>) and run again. Speakers, noises and sound effects below are still real results.</div>}
        </div>
      )}
      {(a.warnings ?? []).map(w => <div key={w} className="banner banner-yellow">{w}</div>)}
      {a.asr.status === 'ok' && a.asr.warnings.length > 0 && <div className="banner banner-yellow">{a.asr.warnings.join(' ')}</div>}

      <div className="summary-row">
        <Stat label="Situation" value={a.asr.status === 'ok' ? sc.label : 'Needs transcript'} sub={sc.confidence ? `${pct(sc.confidence)} sure${sc.evidence.length ? ' · ' + sc.evidence.slice(0, 3).join(', ') : ''}` : undefined} />
        <Stat label="Speakers" value={String(a.diarization.n_speakers)} sub={`${a.diarization.backend} voice model`} />
        <Stat label="Speech" value={pct(a.scene.stats.speech?.share)} sub={`${a.scene.stats.speech?.seconds ?? 0}s of ${a.duration}s`} tone="ok" />
        <Stat label="Sound effects" value={String(a.scene.segments.filter(s => s.state === 'sound_effect').length)} sub="events" tone="alert" />
        <Stat label="Background noise" value={pct(a.scene.stats.background_noise?.share)} sub="of the recording" tone="warn" />
        <Stat label="Tasks found" value={String(a.nlu.tasks.length)} sub={a.nlu.tasks.length ? 'assigned to speakers' : 'none heard'} />
      </div>

      {mediaUrl && (isVideo
        ? <video ref={r => { media.current = r }} className="player" src={mediaUrl} controls />
        : <audio ref={r => { media.current = r }} className="player" src={mediaUrl} controls />)}
      <Timeline blocks={blocks} duration={a.duration} onSelect={select} selectedId={sel?.id} />
      {sel && (
        <div className="selected-card">
          <strong>{sel.category === 'speech' ? (sel.speaker ?? 'Speech') : sel.label}</strong> · {CATEGORY_NAME[sel.category]} · {mmss(sel.start)}–{mmss(sel.end)}
          {sel.confidence != null && <> · {pct(sel.confidence)} confidence</>}
          {sel.text && <p>{sel.text}</p>}
        </div>
      )}

      <div className="export-row">
        <span className="muted">Export</span>
        {(['srt', 'vtt', 'txt', 'csv', 'json'] as const).map(f => <button key={f} className="btn btn-ghost" onClick={() => void exportTimeline(f, a)}>{f.toUpperCase()}</button>)}
      </div>

      <nav className="subtabs" role="tablist">
        {TABS.map(t => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t}{t === 'Tasks' && a.nlu.tasks.length ? <b>{a.nlu.tasks.length}</b> : null}</button>)}
      </nav>
      <div className="subpanel">
        {tab === 'Speakers' && <SpeakersPanel a={a} seek={seek} />}
        {tab === 'Tasks' && <TasksPanel a={a} />}
        {tab === 'Sounds & noise' && <SoundsPanel a={a} />}
        {tab === 'MFCC & spectrogram' && <SpectralPanel a={a} />}
        {tab === 'HMM' && <HmmPanel a={a} />}
        {tab === 'Voice patterns' && <PatternsPanel a={a} />}
      </div>
    </section>
  )
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: 'ok' | 'alert' | 'warn' }) {
  return <div className={`stat ${tone ? 'tone-' + tone : ''}`}><span>{label}</span><strong>{value}</strong>{sub && <small>{sub}</small>}</div>
}

/* ------------------------------------------------------------------ Speakers */
function SpeakersPanel({ a, seek }: { a: Analysis; seek: (t: number) => void }) {
  if (!a.nlu.speakers.length) return <Empty a={a} what="what each person talked about" />
  return (
    <div>
      <div className="cards">
        {a.nlu.speakers.map(p => (
          <article key={p.speaker_id} className="card" style={{ borderTopColor: speakerColor(p.speaker_id) }}>
            <header><h3>{p.label}</h3><span className="chip">{pct(p.share)} of talk time</span></header>
            <p className="lead">{p.llm_summary ?? p.summary}</p>
            <div className="tags">{p.topics.map(t => <span key={t} className="tag">{t}</span>)}</div>
            <dl className="kv"><dt>Spoke for</dt><dd>{p.talk_seconds}s in {p.turns} turn(s)</dd><dt>Pace</dt><dd>{p.words_per_minute || '–'} words/min</dd></dl>
            {p.key_points.length > 0 && <><h4>Key points</h4><ul>{p.key_points.map((k, i) => <li key={i}><button className="link" title="Play from here" onClick={() => seek(k.t)}>{mmss(k.t)}</button> {k.kind === 'decision' && <span className="chip chip-yellow">decision</span>} {k.text}</li>)}</ul></>}
            {p.questions.length > 0 && <><h4>Asked</h4><ul>{p.questions.map((q, i) => <li key={i}>{q}</li>)}</ul></>}
          </article>
        ))}
      </div>
      <h3 className="section">Conversation</h3>
      <div className="transcript">
        {a.utterances.map((u, i) => (
          <div key={i} className="utt" onClick={() => seek(u.start)} title="Play from here"><span className="utt-t">{mmss(u.start)}</span><span className="utt-who" style={{ color: speakerColor(u.speaker_id) }}>{u.speaker}</span><span className="utt-text">{u.text}{u.delivery_style && u.delivery_style !== 'Steady Statement' && <em className="chip">{u.delivery_style}</em>}</span></div>
        ))}
      </div>
    </div>
  )
}

function Empty({ a, what }: { a: Analysis; what: string }) {
  return <p className="muted empty">{a.asr.status === 'ok' ? 'Nothing was said that could be analysed.' : `Needs a transcript to work out ${what}. Speech-to-text is not available (see the red notice above).`}</p>
}

/* ------------------------------------------------------------------ Tasks */
const PRI: Record<string, string> = { high: 'st-alert', medium: 'st-warn', low: 'st-idle' }

function TasksPanel({ a }: { a: Analysis }) {
  const [stored, setStored] = useState<StoredTask[]>([])
  useEffect(() => { if (a.session_id != null) void listTasks(a.session_id).then(setStored).catch(() => setStored([])) }, [a.session_id])
  const done = (t: Task) => stored.find(s => s.title === t.task)?.status === 'completed'
  const toggle = async (t: Task) => {
    const s = stored.find(x => x.title === t.task)
    if (!s) return
    const upd = await setTaskStatus(s.id, s.status === 'completed' ? 'pending' : 'completed')
    setStored(prev => prev.map(x => (x.id === upd.id ? upd : x)))
  }
  if (!a.nlu.tasks.length) return <p className="muted empty">{a.asr.status === 'ok' ? 'No one committed to or was asked to do anything in this recording.' : 'Tasks come from what people say, so they need a transcript. Speech-to-text is not available (see the red notice above).'}</p>
  const owners = [...new Set(a.nlu.tasks.map(t => t.owner))]
  return (
    <div>
      {owners.map(o => {
        const mine = a.nlu.tasks.filter(t => t.owner === o)
        const sid = mine[0].owner_id
        return (
          <section key={o} className="owner-block">
            <h3><i className="dot" style={{ background: speakerColor(sid) }} />{o}<span className="chip">{mine.length} task{mine.length > 1 ? 's' : ''}</span></h3>
            <ul className="tasks">
              {mine.map(t => (
                <li key={t.id} className={done(t) ? 'done' : ''}>
                  <input type="checkbox" checked={done(t)} disabled={!stored.some(s => s.title === t.task)} onChange={() => void toggle(t)} aria-label={`Mark "${t.task}" done`} />
                  <div>
                    <strong>{t.task}</strong>
                    <div className="meta">
                      {t.due && <span className="chip chip-yellow">due {t.due}</span>}
                      <span className={`chip ${PRI[t.priority]}`}>{t.priority}</span>
                      {t.confirmed && <span className="chip st-ok">confirmed</span>}
                      <span className="muted">said by {t.said_by} at {mmss(t.start)}{t.requested_by ? ` · asked by ${t.requested_by}` : ''}</span>
                    </div>
                    {t.source_text && <blockquote>“{t.source_text}”</blockquote>}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )
      })}
      {a.nlu.decisions.length > 0 && <><h3 className="section">Decisions</h3><ul>{a.nlu.decisions.map((d, i) => <li key={i}>{d.speaker} · {mmss(d.t)} — {d.text}</li>)}</ul></>}
    </div>
  )
}

/* ------------------------------------------------------------------ Sounds & noise */
function SoundsPanel({ a }: { a: Analysis }) {
  const segs = a.scene.segments.filter(s => s.state !== 'speech' && s.state !== 'silence')
  return (
    <div>
      <div className="share-bar" aria-label="Share of time by state">
        {(['speech', 'sound_effect', 'background_noise', 'silence'] as const).map(k => {
          const s = a.scene.stats[k]
          return s && s.share > 0 ? <div key={k} className={`share share-${k}`} style={{ flex: s.share }} title={`${CATEGORY_NAME[k]} ${s.seconds}s`}>{s.share > 0.07 ? `${CATEGORY_NAME[k]} ${pct(s.share)}` : ''}</div> : null
        })}
      </div>
      {segs.length === 0 ? <p className="muted empty">No sound effects or background noise were detected outside of speech.</p> : (
        <table className="table">
          <thead><tr><th>When</th><th>What</th><th>Type</th><th>Confidence</th></tr></thead>
          <tbody>{segs.map(s => (
            <tr key={s.id}><td>{mmss(s.start)} – {mmss(s.end)}</td><td><strong>{s.label}</strong></td>
              <td><span className={`chip ${s.state === 'sound_effect' ? 'st-alert' : 'st-warn'}`}>{CATEGORY_NAME[s.state]}</span></td><td>{pct(s.confidence)}</td></tr>
          ))}</tbody>
        </table>
      )}
      <p className="note">Sound effects are labelled from ten trained classes (clapping, coughing, dog, door knock, footsteps, glass breaking, laughing, rain, siren, sneezing). Anything else steady and non-speech is reported as background noise, with a description such as “steady low-frequency hum”. Label accuracy for effects is limited; see the README for measured numbers.</p>
    </div>
  )
}

/* ------------------------------------------------------------------ MFCC & spectrogram */
function SpectralPanel({ a }: { a: Analysis }) {
  const s = a.patterns.spectral
  if (!s.mfcc) return <p className="muted empty">Recording too short for spectral analysis.</p>
  return (
    <div>
      {s.covered_seconds < a.duration && <p className="note">Showing the first {s.covered_seconds}s of {a.duration}s.</p>}
      <h3 className="section">MFCC (13 coefficients over time)</h3>
      <Heatmap data={s.mfcc} height={190} xLabel="time →" yLabel="coef" duration={s.covered_seconds} />
      <h3 className="section">Spectrogram (STFT, dB)</h3>
      <Heatmap data={s.stft_db ?? []} flipY height={240} xLabel="time →" yLabel="Hz" duration={s.covered_seconds} yMaxLabel={`${(s.stft_freq_max_hz ?? 8000) / 1000} kHz`} />
      <div className="grid2">
        <LineChart values={s.rms ?? []} label="Energy (RMS)" fill color="var(--navy)" />
        <LineChart values={s.f0_contour_hz ?? []} label="Pitch (F0)" unit="Hz" color="var(--amber)" min={0} />
      </div>
      <div className="grid2">
        <LineChart values={s.lpc_envelope_db ?? []} label="LPC spectral envelope (loudest frame)" unit="dB" color="var(--navy)" fill />
        <div className="kv-block">
          <dl className="kv"><dt>Formants</dt><dd>{(s.lpc_formants_hz ?? []).length ? s.lpc_formants_hz!.map(f => `${f} Hz`).join(', ') : '–'}</dd>
            <dt>Mean pitch</dt><dd>{s.f0_mean_hz ? `${s.f0_mean_hz} Hz` : '–'}</dd>
            <dt>Voiced frames</dt><dd>{pct(s.voiced_ratio)}</dd>
            <dt>Noise reduction</dt><dd>{s.snr_before_db != null ? `SNR ${s.snr_before_db} → ${s.snr_after_db} dB (${(s.snr_gain_db ?? 0) >= 0 ? '+' : ''}${s.snr_gain_db} dB)` : '–'}</dd></dl>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ HMM */
const STATE_COLOR: Record<string, string> = { speech: 'var(--green)', sound_effect: 'var(--red)', background_noise: 'var(--amber)', silence: 'var(--grey)' }
const SYM_COLOR: Record<string, string> = { speech: 'var(--green)', effect: 'var(--red)', noise: 'var(--amber)', silence: 'var(--grey)' }

function Strip({ items, colors, title }: { items: string[]; colors: Record<string, string>; title: string }) {
  return (
    <div className="strip-wrap">
      <span className="strip-title">{title}</span>
      <div className="strip">{items.map((s, i) => <i key={i} style={{ background: colors[s] ?? 'var(--grey)' }} title={`#${i}: ${s}`} />)}</div>
    </div>
  )
}

function Matrix({ title, rows, cols, m }: { title: string; rows: string[]; cols: string[]; m: number[][] }) {
  return (
    <div className="matrix">
      <h4>{title}</h4>
      <table className="table compact"><thead><tr><th />{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>{m.map((r, i) => <tr key={i}><th>{rows[i]}</th>{r.map((v, j) => <td key={j} style={{ background: `rgba(37,99,235,${Math.min(0.85, v)})`, color: v > 0.45 ? '#fff' : 'inherit' }}>{v.toFixed(2)}</td>)}</tr>)}</tbody></table>
    </div>
  )
}

function HmmPanel({ a }: { a: Analysis }) {
  const h = a.patterns.hmm.scene
  if (!h.model) return <p className="muted empty">No HMM data.</p>
  const changed = h.observations.reduce((n, o, i) => n + (o !== ({ speech: 'speech', sound_effect: 'effect', background_noise: 'noise', silence: 'silence' } as Record<string, string>)[h.path[i]] ? 1 : 0), 0)
  return (
    <div>
      <p>Every {h.hop_s}s the classifier makes a raw decision. A four-state HMM (silence · speech · sound effect · background noise) then picks the single most likely sequence with the Viterbi algorithm, which removes one-hop flicker.</p>
      <Strip items={h.observations} colors={SYM_COLOR} title="Raw per-hop decisions" />
      <Strip items={h.path} colors={STATE_COLOR} title="After Viterbi smoothing" />
      <p className="note">{changed} of {h.observations.length} hops were changed by smoothing · log-likelihood {h.log_likelihood?.toFixed(1) ?? '–'} · best-path log-probability {h.best_path_log_prob?.toFixed(1) ?? '–'}</p>
      <div className="grid2">
        <Matrix title="Transition probabilities A" rows={h.model.states} cols={h.model.states} m={h.model.A} />
        <Matrix title="Emission probabilities B" rows={h.model.states} cols={h.model.symbols} m={h.model.B} />
      </div>
      {a.patterns.hmm.prosody && (
        <>
          <h3 className="section">Conversation-flow HMM (every {a.patterns.hmm.prosody.step_s.toFixed(1)}s)</h3>
          <p className="note">States inferred from how the speech sounds (pauses, rising pitch, stress). Hidden states: turn pause, explanation, emphasis, question.</p>
          <div className="path-list">{a.patterns.hmm.prosody.best_path.map((s, i) => <span key={i} className="chip">{s}</span>)}</div>
        </>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ Voice patterns */
function PatternsPanel({ a }: { a: Analysis }) {
  const pts = a.diarization.projection
  const sp = a.patterns.speakers
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y)
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  return (
    <div>
      <h3 className="section">Who sounds like whom</h3>
      <p className="note">Each dot is a short stretch of speech, plotted by voice similarity ({a.diarization.backend} speaker embedding, 2-D projection). Dots that sit together are the same voice.</p>
      {pts.length > 0 ? (
        <svg viewBox="0 0 600 260" className="scatter">
          {pts.map((p, i) => <circle key={i} cx={20 + ((p.x - x0) / (x1 - x0 || 1)) * 560} cy={20 + ((p.y - y0) / (y1 - y0 || 1)) * 220} r={5} fill={speakerColor(p.speaker_id)} opacity={0.8}><title>{`Speaker ${p.speaker_id} @ ${mmss(p.t)}`}</title></circle>)}
        </svg>
      ) : <p className="muted">Not enough speech to plot.</p>}
      <div className="tags">{a.speakers.map(s => <span key={s.speaker_id} className="tag" style={{ borderColor: speakerColor(s.speaker_id) }}><i className="dot" style={{ background: speakerColor(s.speaker_id) }} />{s.label}</span>)}</div>
      <h3 className="section">Per-speaker voice profile</h3>
      <div className="cards">
        {sp.map(p => (
          <article key={p.speaker_id} className="card" style={{ borderTopColor: speakerColor(p.speaker_id) }}>
            <header><h3>{p.speaker}</h3></header>
            <dl className="kv"><dt>Mean pitch</dt><dd>{p.mean_f0_hz || '–'} Hz</dd><dt>Pitch variation</dt><dd>±{p.f0_std_hz} Hz</dd><dt>Voiced</dt><dd>{pct(p.voiced_ratio)}</dd><dt>Analysed</dt><dd>{p.seconds_analysed}s</dd></dl>
            <h4>Average MFCC</h4>
            <Bars values={p.mfcc_mean.slice(1)} labels={p.mfcc_mean.slice(1).map((_, i) => `c${i + 1}`)} color={speakerColor(p.speaker_id)} />
          </article>
        ))}
      </div>
    </div>
  )
}
