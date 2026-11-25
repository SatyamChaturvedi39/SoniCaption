import { useEffect, useRef } from 'react'

function heatColor(norm: number): string {
  const t = Math.max(0, Math.min(1, norm))
  const r = Math.round(15 + t * 235)
  const g = Math.round(23 + Math.sin(t * Math.PI) * 140 + t * 40)
  const b = Math.round(70 + (1 - t) * 145)
  return `rgb(${r},${g},${b})`
}

export function SpectrogramCanvas({ matrix, title }: { matrix: number[][]; title?: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !matrix.length || !matrix[0]?.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const rows = matrix.length
    const cols = matrix[0].length
    const flat = matrix.flat()
    const mn = Math.min(...flat)
    const mx = Math.max(...flat)
    const span = Math.max(mx - mn, 1e-5)
    const cellW = canvas.width / cols
    const cellH = canvas.height / rows
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = (matrix[rows - 1 - r][c] - mn) / span
        ctx.fillStyle = heatColor(val)
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH))
      }
    }
  }, [matrix])

  return (
    <div className="canvas-wrap">
      {title && <div className="canvas-title">{title}</div>}
      <canvas ref={ref} width={560} height={190} className="lab-canvas" />
    </div>
  )
}

export function DualLineCanvas({
  primary,
  secondary,
  labelA = 'Raw Signal',
  labelB = 'Denoised',
}: {
  primary: number[]
  secondary?: number[]
  labelA?: string
  labelB?: string
}) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !primary.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const { width: w, height: h } = canvas
    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#0b1324'
    ctx.fillRect(0, 0, w, h)

    const all = secondary?.length ? [...primary, ...secondary] : primary
    const absMax = Math.max(0.01, ...all.map((v) => Math.abs(v)))

    const drawSeries = (arr: number[], color: string, alpha: number) => {
      ctx.save()
      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()
      arr.forEach((v, i) => {
        const x = (i / Math.max(arr.length - 1, 1)) * w
        const y = h / 2 - (v / absMax) * (h * 0.42)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      ctx.restore()
    }
    drawSeries(primary, '#f87171', 0.55)
    if (secondary?.length) drawSeries(secondary, '#38bdf8', 0.95)
  }, [primary, secondary])

  return (
    <div className="canvas-wrap">
      <div className="canvas-legend">
        <span><i style={{ background: '#f87171' }} /> {labelA}</span>
        {secondary?.length ? <span><i style={{ background: '#38bdf8' }} /> {labelB}</span> : null}
      </div>
      <canvas ref={ref} width={560} height={150} className="lab-canvas" />
    </div>
  )
}

export function DtwCanvas({ matrix, path }: { matrix: number[][]; path: number[][] }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !matrix.length || !matrix[0]?.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const rows = matrix.length
    const cols = matrix[0].length
    const flat = matrix.flat()
    const mn = Math.min(...flat)
    const mx = Math.max(...flat)
    const span = Math.max(mx - mn, 1e-5)
    const cellW = canvas.width / cols
    const cellH = canvas.height / rows

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const norm = (matrix[r][c] - mn) / span
        const shade = Math.round(25 + norm * 190)
        ctx.fillStyle = `rgb(${shade}, ${Math.round(50 + norm * 80)}, ${130})`
        ctx.fillRect(c * cellW, (rows - 1 - r) * cellH, Math.ceil(cellW), Math.ceil(cellH))
      }
    }

    if (path?.length) {
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 3
      ctx.beginPath()
      path.forEach(([r, c], i) => {
        const x = (c + 0.5) * cellW
        const y = (rows - 1 - r + 0.5) * cellH
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
    }
  }, [matrix, path])

  return (
    <div className="canvas-wrap">
      <div className="canvas-title">Accumulated DTW Cost Matrix + Optimal Warp Path (Gold)</div>
      <canvas ref={ref} width={440} height={260} className="lab-canvas" />
    </div>
  )
}

export function HmmTrellisSvg({ data }: { data: any }) {
  const nodes: any[] = data?.trellis_nodes || []
  const edges: any[] = data?.trellis_edges || []
  const states: string[] = data?.model?.states || []
  const obs: string[] = data?.observations || []
  if (!nodes.length || !states.length) return null

  const colW = Math.max(85, Math.min(130, 680 / Math.max(obs.length, 1)))
  const width = Math.max(420, obs.length * colW + 90)
  const height = states.length * 76 + 70
  const pos = (step: number, state: string) => ({
    x: 70 + step * colW,
    y: 45 + states.indexOf(state) * 76,
  })

  return (
    <div className="trellis-scroll">
      <svg width={width} height={height} className="trellis-svg">
        {edges.map((e, i) => {
          const a = pos(e.step - 1, e.from)
          const b = pos(e.step, e.to)
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={e.chosen ? '#fbbf24' : '#334155'}
              strokeWidth={e.chosen ? 3 : 1}
              strokeOpacity={e.chosen ? 1 : 0.45}
            />
          )
        })}
        {obs.map((o, t) => (
          <text key={`o-${t}`} x={70 + t * colW} y={height - 12} fill="#94a3b8" fontSize="11" textAnchor="middle">
            t{t + 1}: {o}
          </text>
        ))}
        {nodes.map((n, i) => {
          const p = pos(n.step, n.state)
          return (
            <g key={i}>
              <circle
                cx={p.x}
                cy={p.y}
                r={18}
                fill={n.on_best_path ? '#0ea5e9' : '#1e293b'}
                stroke={n.on_best_path ? '#fbbf24' : '#475569'}
                strokeWidth={n.on_best_path ? 2.5 : 1.5}
              />
              <text x={p.x} y={p.y - 1} fill="#f8fafc" fontSize="10" fontWeight="700" textAnchor="middle">
                {n.state}
              </text>
              <text x={p.x} y={p.y + 11} fill="#cbd5e1" fontSize="8.5" textAnchor="middle">
                {n.log_delta != null ? n.log_delta.toFixed(1) : '-∞'}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
