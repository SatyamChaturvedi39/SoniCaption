import { useEffect, useRef } from 'react'

function cyberHeatColor(norm: number): string {
  const t = Math.max(0, Math.min(1, norm))
  // Cyber viridis/electric audio gradient:
  // t=0: dark navy (#090918)
  // t=0.35: deep purple (#4c1d95)
  // t=0.7: electric cyan/emerald (#06b6d4 / #10b981)
  // t=1.0: bright warm yellow-white (#fef08a)
  let r: number, g: number, b: number
  if (t < 0.25) {
    const sub = t / 0.25
    r = Math.round(9 + sub * 25)
    g = Math.round(9 + sub * 15)
    b = Math.round(24 + sub * 85)
  } else if (t < 0.6) {
    const sub = (t - 0.25) / 0.35
    r = Math.round(34 + sub * 40)
    g = Math.round(24 + sub * 130)
    b = Math.round(109 + sub * 90)
  } else if (t < 0.85) {
    const sub = (t - 0.6) / 0.25
    r = Math.round(74 + sub * 130)
    g = Math.round(154 + sub * 80)
    b = Math.round(199 - sub * 80)
  } else {
    const sub = (t - 0.85) / 0.15
    r = Math.round(204 + sub * 51)
    g = Math.round(234 + sub * 21)
    b = Math.round(119 + sub * 136)
  }
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
        ctx.fillStyle = cyberHeatColor(val)
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
    ctx.fillStyle = '#08080d'
    ctx.fillRect(0, 0, w, h)

    // Subtle background grid lines
    ctx.strokeStyle = '#181826'
    ctx.lineWidth = 1
    for (let x = 0; x < w; x += 50) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
      ctx.stroke()
    }
    for (let y = 0; y < h; y += 35) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
      ctx.stroke()
    }

    const all = secondary?.length ? [...primary, ...secondary] : primary
    const absMax = Math.max(0.01, ...all.map((v) => Math.abs(v)))

    const drawSeries = (arr: number[], color: string, alpha: number) => {
      ctx.save()
      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
      ctx.lineWidth = 1.9
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
    drawSeries(primary, '#f43f5e', 0.65)
    if (secondary?.length) drawSeries(secondary, '#06b6d4', 0.95)
  }, [primary, secondary])

  return (
    <div className="canvas-frame">
      <div className="canvas-legend">
        <span>
          <i style={{ background: '#f43f5e' }} /> {labelA}
        </span>
        {secondary?.length ? (
          <span>
            <i style={{ background: '#06b6d4' }} /> {labelB}
          </span>
        ) : null}
      </div>
      <canvas ref={ref} width={560} height={145} className="lab-canvas" />
    </div>
  )
}
