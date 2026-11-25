import { useEffect, useRef } from 'react'

function heatColor(norm: number): string {
  const t = Math.max(0, Math.min(1, norm))
  const r = Math.round(18 + t * 225)
  const g = Math.round(28 + Math.sin(t * Math.PI) * 125 + t * 40)
  const b = Math.round(52 + (1 - t) * 75)
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
    <div className="canvas-frame">
      {title && <div className="canvas-title">{title}</div>}
      <canvas ref={ref} width={560} height={170} className="lab-canvas" />
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
    ctx.fillStyle = '#14171c'
    ctx.fillRect(0, 0, w, h)

    const all = secondary?.length ? [...primary, ...secondary] : primary
    const absMax = Math.max(0.01, ...all.map((v) => Math.abs(v)))

    const drawSeries = (arr: number[], color: string, alpha: number) => {
      ctx.save()
      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
      ctx.lineWidth = 1.8
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
    drawSeries(primary, '#ea580c', 0.6)
    if (secondary?.length) drawSeries(secondary, '#2dd4bf', 0.95)
  }, [primary, secondary])

  return (
    <div className="canvas-frame">
      <div className="canvas-legend">
        <span><i style={{ background: '#ea580c' }} /> {labelA}</span>
        {secondary?.length ? <span><i style={{ background: '#2dd4bf' }} /> {labelB}</span> : null}
      </div>
      <canvas ref={ref} width={560} height={145} className="lab-canvas" />
    </div>
  )
}
