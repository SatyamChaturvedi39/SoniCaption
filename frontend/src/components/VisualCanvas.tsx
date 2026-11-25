import { useEffect, useRef } from 'react'

/** Scientific Inferno perceptual colormap: silence (deep obsidian) -> purple -> crimson -> amber -> bright yellow peaks */
function infernoColor(norm: number): string {
  const t = Math.max(0, Math.min(1, norm))
  const stops: [number, [number, number, number]][] = [
    [0.0, [9, 10, 18]],
    [0.25, [82, 16, 110]],
    [0.5, [182, 54, 85]],
    [0.75, [248, 137, 36]],
    [1.0, [252, 255, 164]],
  ]
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i]
    const [t1, c1] = stops[i + 1]
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0)
      const r = Math.round(c0[0] + f * (c1[0] - c0[0]))
      const g = Math.round(c0[1] + f * (c1[1] - c0[1]))
      const b = Math.round(c0[2] + f * (c1[2] - c0[2]))
      return `rgb(${r},${g},${b})`
    }
  }
  return 'rgb(252,255,164)'
}

/** Teal-Slate-Amber diverging colormap for zero-centered MFCC cepstral coefficients */
function mfccDivergingColor(norm: number): string {
  const t = Math.max(0, Math.min(1, norm))
  if (t < 0.5) {
    const f = t / 0.5
    const r = Math.round(14 + f * (28 - 14))
    const g = Math.round(165 + f * (34 - 165))
    const b = Math.round(233 + f * (52 - 233))
    return `rgb(${r},${g},${b})`
  }
  const f = (t - 0.5) / 0.5
  const r = Math.round(28 + f * (251 - 28))
  const g = Math.round(34 + f * (191 - 34))
  const b = Math.round(52 + f * (36 - 52))
  return `rgb(${r},${g},${b})`
}

export function SpectrogramCanvas({
  matrix,
  title,
  subtitle,
  durationSec = 5,
  maxFreqHz = 8000,
  mode = 'stft',
}: {
  matrix: number[][]
  title?: string
  subtitle?: string
  durationSec?: number
  maxFreqHz?: number
  mode?: 'stft' | 'mfcc'
}) {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !matrix.length || !matrix[0]?.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const padL = 46
    const padR = 46
    const padT = 10
    const padB = 24
    const plotW = W - padL - padR
    const plotH = H - padT - padB

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#090b12'
    ctx.fillRect(0, 0, W, H)

    const rows = matrix.length
    const cols = matrix[0].length

    let mn = -75
    let mx = 0
    if (mode === 'mfcc') {
      const sorted = matrix.flat().slice().sort((a, b) => a - b)
      const p03 = sorted[Math.floor(sorted.length * 0.03)] ?? -1
      const p97 = sorted[Math.floor(sorted.length * 0.97)] ?? 1
      const absLim = Math.max(Math.abs(p03), Math.abs(p97), 1)
      mn = -absLim
      mx = absLim
    } else {
      const flat = matrix.flat()
      mx = Math.max(...flat)
      mn = Math.max(Math.min(...flat), mx - 75)
    }
    const span = Math.max(mx - mn, 1e-5)

    const cellW = plotW / cols
    const cellH = plotH / rows

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = (matrix[rows - 1 - r][c] - mn) / span
        ctx.fillStyle = mode === 'mfcc' ? mfccDivergingColor(val) : infernoColor(val)
        ctx.fillRect(padL + c * cellW, padT + r * cellH, Math.ceil(cellW), Math.ceil(cellH))
      }
    }

    // Plot border
    ctx.strokeStyle = '#23293d'
    ctx.lineWidth = 1
    ctx.strokeRect(padL, padT, plotW, plotH)

    // Y-axis ticks
    ctx.fillStyle = '#94a3b8'
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'

    if (mode === 'stft') {
      const ySteps = 4
      for (let i = 0; i <= ySteps; i++) {
        const frac = i / ySteps
        const y = padT + (1 - frac) * plotH
        const hz = (frac * maxFreqHz) / 1000
        ctx.fillText(hz === 0 ? '0Hz' : `${hz.toFixed(0)}k`, padL - 6, y)
        if (i > 0 && i < ySteps) {
          ctx.strokeStyle = 'rgba(255,255,255,0.08)'
          ctx.beginPath()
          ctx.moveTo(padL, y)
          ctx.lineTo(padL + plotW, y)
          ctx.stroke()
        }
      }
    } else {
      ctx.fillText(`c${rows}`, padL - 6, padT + 6)
      ctx.fillText(`c${Math.ceil(rows / 2)}`, padL - 6, padT + plotH / 2)
      ctx.fillText('c1', padL - 6, padT + plotH - 5)
    }

    // X-axis time ticks
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    const dur = durationSec > 0 ? durationSec : 5
    const xTicks = 4
    for (let i = 0; i <= xTicks; i++) {
      const frac = i / xTicks
      const x = padL + frac * plotW
      ctx.fillText(`${(frac * dur).toFixed(1)}s`, x, padT + plotH + 6)
    }

    // Vertical colorbar on right
    const barX = padL + plotW + 10
    const barW = 10
    for (let yPix = 0; yPix < plotH; yPix++) {
      const norm = 1 - yPix / plotH
      ctx.fillStyle = mode === 'mfcc' ? mfccDivergingColor(norm) : infernoColor(norm)
      ctx.fillRect(barX, padT + yPix, barW, 1.5)
    }
    ctx.strokeStyle = '#23293d'
    ctx.strokeRect(barX, padT, barW, plotH)
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillStyle = '#94a3b8'
    if (mode === 'stft') {
      ctx.fillText('0dB', barX + 13, padT + 6)
      ctx.fillText('-75', barX + 13, padT + plotH - 5)
    } else {
      ctx.fillText('+', barX + 13, padT + 6)
      ctx.fillText('0', barX + 13, padT + plotH / 2)
      ctx.fillText('-', barX + 13, padT + plotH - 5)
    }
  }, [matrix, durationSec, maxFreqHz, mode])

  return (
    <div className="canvas-frame">
      {title && <div className="canvas-title" style={{ marginBottom: subtitle ? 2 : 8 }}>{title}</div>}
      {subtitle && <div className="muted" style={{ fontSize: 11, marginBottom: 8 }}>{subtitle}</div>}
      <canvas ref={ref} width={580} height={200} className="lab-canvas" />
    </div>
  )
}

export function DualLineCanvas({
  primary,
  secondary,
  labelA = 'Raw Signal',
  labelB = 'Denoised',
  subtitle,
  mode = 'waveform',
  durationSec = 5,
  maxFreqHz = 8000,
  formantsHz = [],
}: {
  primary: number[]
  secondary?: number[]
  labelA?: string
  labelB?: string
  subtitle?: string
  mode?: 'waveform' | 'spectrum'
  durationSec?: number
  maxFreqHz?: number
  formantsHz?: number[]
}) {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !primary.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const padL = 46
    const padR = 14
    const padT = 14
    const padB = 24
    const plotW = W - padL - padR
    const plotH = H - padT - padB

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#090b12'
    ctx.fillRect(0, 0, W, H)

    const all = secondary?.length ? [...primary, ...secondary] : primary
    let yMin = -1
    let yMax = 1
    if (mode === 'spectrum') {
      yMin = -72
      yMax = 4
    } else {
      const absMax = Math.max(0.05, ...all.map((v) => Math.abs(v)))
      yMin = -absMax
      yMax = absMax
    }
    const ySpan = Math.max(yMax - yMin, 1e-4)
    const mapY = (v: number) => {
      const clamped = Math.max(yMin, Math.min(yMax, v))
      return padT + (1 - (clamped - yMin) / ySpan) * plotH
    }

    // Horizontal grid & Y ticks
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'

    const yTicks = mode === 'spectrum' ? [0, -20, -40, -60] : [yMax, 0, yMin]
    for (const val of yTicks) {
      const y = mapY(val)
      ctx.strokeStyle = val === 0 && mode === 'waveform' ? 'rgba(148,163,184,0.35)' : 'rgba(255,255,255,0.07)'
      ctx.setLineDash(val === 0 ? [4, 3] : [])
      ctx.beginPath()
      ctx.moveTo(padL, y)
      ctx.lineTo(padL + plotW, y)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#94a3b8'
      const lbl = mode === 'spectrum' ? `${val}dB` : val === 0 ? '0' : val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)
      ctx.fillText(lbl, padL - 6, y)
    }

    // X-axis ticks
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    const xTicks = 4
    for (let i = 0; i <= xTicks; i++) {
      const frac = i / xTicks
      const x = padL + frac * plotW
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.beginPath()
      ctx.moveTo(x, padT)
      ctx.lineTo(x, padT + plotH)
      ctx.stroke()
      ctx.fillStyle = '#94a3b8'
      const lbl = mode === 'spectrum' ? `${((frac * maxFreqHz) / 1000).toFixed(0)}kHz` : `${(frac * durationSec).toFixed(1)}s`
      ctx.fillText(lbl, x, padT + plotH + 6)
    }

    const drawSeries = (arr: number[], color: string, alpha: number, width: number) => {
      ctx.save()
      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.beginPath()
      arr.forEach((v, i) => {
        const x = padL + (i / Math.max(arr.length - 1, 1)) * plotW
        const y = mapY(v)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      ctx.restore()
    }

    drawSeries(primary, '#f97316', mode === 'spectrum' ? 0.72 : 0.65, 1.5)
    if (secondary?.length) {
      drawSeries(secondary, '#22d3ee', 0.98, mode === 'spectrum' ? 2.2 : 1.7)
    }

    // Formant resonance pins (F1, F2, F3) in Spectrum mode
    if (mode === 'spectrum' && formantsHz.length > 0 && maxFreqHz > 0) {
      formantsHz.slice(0, 3).forEach((hz, idx) => {
        if (hz <= 0 || hz > maxFreqHz) return
        const x = padL + (hz / maxFreqHz) * plotW
        ctx.save()
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.75)'
        ctx.setLineDash([3, 3])
        ctx.lineWidth = 1.2
        ctx.beginPath()
        ctx.moveTo(x, padT + 14)
        ctx.lineTo(x, padT + plotH)
        ctx.stroke()
        ctx.fillStyle = '#10b981'
        ctx.font = 'bold 9.5px "JetBrains Mono", monospace'
        ctx.textAlign = 'center'
        ctx.fillText(`F${idx + 1}:${Math.round(hz)}`, x, padT + 4)
        ctx.restore()
      })
    }

    // Frame outline
    ctx.strokeStyle = '#23293d'
    ctx.lineWidth = 1
    ctx.strokeRect(padL, padT, plotW, plotH)
  }, [primary, secondary, mode, durationSec, maxFreqHz, formantsHz])

  return (
    <div className="canvas-frame">
      <div className="canvas-legend" style={{ marginBottom: subtitle ? 2 : 8 }}>
        <span><i style={{ background: '#f97316' }} /> {labelA}</span>
        {secondary?.length ? <span><i style={{ background: '#22d3ee' }} /> {labelB}</span> : null}
      </div>
      {subtitle && <div className="muted" style={{ fontSize: 11, marginBottom: 8 }}>{subtitle}</div>}
      <canvas ref={ref} width={580} height={185} className="lab-canvas" />
    </div>
  )
}
