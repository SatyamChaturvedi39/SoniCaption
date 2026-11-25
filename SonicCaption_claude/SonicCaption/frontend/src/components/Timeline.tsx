import { useMemo, useState } from 'react'
import type { Category, Status } from '../types'
import { CATEGORY_NAME, mmss, speakerColor } from '../lib/fmt'

export interface Block {
  id: string
  start: number
  end: number
  category: Category
  label: string
  status: Status
  speaker_id?: number | null
  speaker?: string | null
  text?: string
  confidence?: number | null
  open?: boolean
}

interface Props {
  blocks: Block[]
  duration: number
  follow?: boolean
  onSelect?: (b: Block) => void
  selectedId?: string | null
}

const LANE_H = 30
const FILL: Record<Status, string> = { ok: 'var(--green)', alert: 'var(--red)', warn: 'var(--amber)', idle: 'var(--grey)' }

/** Swim-lane timeline: one lane per speaker, one for sound effects, one for background noise. */
export default function Timeline({ blocks, duration, follow, onSelect, selectedId }: Props) {
  const [pps, setPps] = useState(40)
  const speakers = useMemo(() => [...new Set(blocks.filter(b => b.category === 'speech').map(b => b.speaker_id ?? 0))].sort((a, b) => a - b), [blocks])
  const lanes = useMemo(() => [
    ...speakers.map(s => ({ key: `spk${s}`, label: s ? `Speaker ${s}` : 'Speaking…', color: speakerColor(s) })),
    { key: 'sound_effect', label: 'Sound effects', color: 'var(--red)' },
    { key: 'background_noise', label: 'Background noise', color: 'var(--amber)' },
  ], [speakers])
  const laneOf = (b: Block): string => (b.category === 'speech' ? `spk${b.speaker_id ?? 0}` : b.category)
  const width = Math.max(600, duration * pps + 40)
  const height = lanes.length * LANE_H + 26
  const ticks: number[] = []
  const step = pps >= 60 ? 1 : pps >= 25 ? 2 : pps >= 12 ? 5 : 10
  for (let t = 0; t <= duration + step; t += step) ticks.push(t)

  return (
    <div className="timeline">
      <div className="timeline-tools">
        <span className="muted">Zoom</span>
        <input type="range" min={8} max={140} value={pps} onChange={e => setPps(Number(e.target.value))} aria-label="Timeline zoom" />
        <Legend />
      </div>
      <div className="timeline-body">
        <div className="lane-labels" style={{ paddingTop: 22 }}>
          {lanes.map(l => <div key={l.key} className="lane-label" style={{ height: LANE_H }}><i style={{ background: l.color }} />{l.label}</div>)}
        </div>
        <div className="timeline-scroll" ref={el => { if (el && follow) el.scrollLeft = el.scrollWidth }}>
          <svg width={width} height={height} role="img" aria-label="Timeline of speech, sounds and noise">
            {ticks.map(t => (
              <g key={t}>
                <line x1={t * pps} x2={t * pps} y1={18} y2={height} className="grid" />
                <text x={t * pps + 3} y={12} className="tick">{mmss(t).replace(/\.\d$/, '')}</text>
              </g>
            ))}
            {lanes.map((_, i) => <rect key={i} x={0} y={22 + i * LANE_H} width={width} height={LANE_H} className={i % 2 ? 'lane-odd' : 'lane-even'} />)}
            {blocks.filter(b => b.category !== 'silence').map(b => {
              const li = lanes.findIndex(l => l.key === laneOf(b))
              if (li < 0) return null
              const x = b.start * pps
              const w = Math.max(3, (b.end - b.start) * pps)
              const fill = b.category === 'speech' ? lanes[li].color : FILL[b.status]
              const sel = selectedId === b.id
              return (
                <g key={b.id} onClick={() => onSelect?.(b)} className="blk">
                  <rect x={x} y={26 + li * LANE_H} width={w} height={LANE_H - 8} rx={5} fill={fill} opacity={b.open ? 0.65 : 1} stroke={sel ? 'var(--navy-dark)' : 'none'} strokeWidth={2} />
                  {w > 46 && <text x={x + 6} y={26 + li * LANE_H + 14} className="blk-text">{b.category === 'speech' ? (b.text || 'speech') : b.label}</text>}
                  <title>{`${CATEGORY_NAME[b.category]}${b.speaker ? ` · ${b.speaker}` : ''}\n${b.label}\n${mmss(b.start)} – ${mmss(b.end)}${b.confidence != null ? `\nconfidence ${Math.round(b.confidence * 100)}%` : ''}${b.text ? `\n${b.text}` : ''}`}</title>
                </g>
              )
            })}
          </svg>
        </div>
      </div>
    </div>
  )
}

export function Legend() {
  return (
    <span className="legend">
      <span><i style={{ background: 'var(--green)' }} />Speech</span>
      <span><i style={{ background: 'var(--red)' }} />Sound effect</span>
      <span><i style={{ background: 'var(--amber)' }} />Background noise</span>
    </span>
  )
}
