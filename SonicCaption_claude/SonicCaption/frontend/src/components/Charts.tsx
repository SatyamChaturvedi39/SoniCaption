import { useEffect, useRef } from 'react'

/** Colour ramp navy -> blue -> light yellow (matches the theme). t in [0,1]. */
function ramp(t: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [[0, [10, 29, 68]], [0.35, [37, 99, 235]], [0.7, [250, 204, 21]], [1, [255, 249, 196]]]
  const x = Math.min(1, Math.max(0, t))
  for (let i = 1; i < stops.length; i++) {
    if (x <= stops[i][0]) {
      const [t0, c0] = stops[i - 1]
      const [t1, c1] = stops[i]
      const u = (x - t0) / (t1 - t0)
      return [c0[0] + (c1[0] - c0[0]) * u, c0[1] + (c1[1] - c0[1]) * u, c0[2] + (c1[2] - c0[2]) * u]
    }
  }
  return stops[stops.length - 1][1]
}

interface HeatProps { data: number[][]; width?: number; height?: number; flipY?: boolean; xLabel?: string; yLabel?: string; duration?: number; yMaxLabel?: string }

/** Matrix heat-map on a canvas. Rows are drawn top-to-bottom unless flipY (low frequency at the bottom). */
export function Heatmap({ data, width = 900, height = 260, flipY, xLabel, yLabel, duration, yMaxLabel }: HeatProps) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || !data.length) return
    const rows = data.length, cols = data[0].length
    let lo = Infinity, hi = -Infinity
    for (const r of data) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v }
    // clip the extremes so one loud frame does not wash out the picture
    const flat = data.flat().sort((a, b) => a - b)
    lo = flat[Math.floor(flat.length * 0.02)] ?? lo
    hi = flat[Math.floor(flat.length * 0.995)] ?? hi
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(cols, rows)
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const [r, g, b] = ramp((data[flipY ? rows - 1 - y : y][x] - lo) / (hi - lo || 1))
      const o = (y * cols + x) * 4
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255
    }
    c.width = cols; c.height = rows
    ctx.putImageData(img, 0, 0)
  }, [data, flipY])
  if (!data.length) return <p className="muted">No data.</p>
  return (
    <figure className="chart">
      <div className="heat-wrap">
        {yLabel && <span className="axis-y">{yLabel}</span>}
        <canvas ref={ref} style={{ width: '100%', maxWidth: width, height, imageRendering: 'pixelated', borderRadius: 8 }} />
      </div>
      <figcaption className="axis-x"><span>0 s</span>{xLabel && <span>{xLabel}</span>}<span>{duration ? `${duration.toFixed(0)} s` : ''}{yMaxLabel ? ` · top = ${yMaxLabel}` : ''}</span></figcaption>
    </figure>
  )
}

interface LineProps { values: number[]; height?: number; color?: string; label?: string; unit?: string; fill?: boolean; min?: number; max?: number }

export function LineChart({ values, height = 110, color = 'var(--navy)', label, unit, fill, min, max }: LineProps) {
  if (!values.length) return <p className="muted">No data.</p>
  const lo = min ?? Math.min(...values), hi = max ?? Math.max(...values)
  const W = 900
  const pts = values.map((v, i) => `${(i / Math.max(1, values.length - 1)) * W},${height - 6 - ((v - lo) / (hi - lo || 1)) * (height - 12)}`)
  return (
    <figure className="chart">
      {label && <figcaption className="chart-title">{label}<span className="muted">{unit ? ` (${unit})` : ''} · {lo.toFixed(2)} … {hi.toFixed(2)}</span></figcaption>}
      <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" className="line-svg" style={{ height }}>
        {fill && <polygon points={`0,${height} ${pts.join(' ')} ${W},${height}`} fill={color} opacity={0.15} />}
        <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth={1.8} vectorEffect="non-scaling-stroke" />
      </svg>
    </figure>
  )
}

export function Bars({ values, labels, color = 'var(--navy)', height = 120 }: { values: number[]; labels?: string[]; color?: string; height?: number }) {
  if (!values.length) return null
  const mx = Math.max(...values.map(Math.abs), 1e-6)
  return (
    <svg viewBox={`0 0 ${values.length * 24} ${height + 16}`} className="bars-svg" style={{ height: height + 16 }}>
      <line x1={0} x2={values.length * 24} y1={height / 2} y2={height / 2} className="grid" />
      {values.map((v, i) => {
        const h = (Math.abs(v) / mx) * (height / 2 - 4)
        return <g key={i}><rect x={i * 24 + 4} y={v >= 0 ? height / 2 - h : height / 2} width={16} height={Math.max(1, h)} rx={2} fill={color} opacity={0.9} /><text x={i * 24 + 12} y={height + 12} className="tick" textAnchor="middle">{labels?.[i] ?? i}</text><title>{`${labels?.[i] ?? i}: ${v.toFixed(2)}`}</title></g>
      })}
    </svg>
  )
}
