import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Pause, Play, Volume2 } from 'lucide-react'

type Props = {
  audioUrl: string
  seekTo?: { time: number; seq: number } | null
  onTimeUpdate?: (time: number) => void
}

const fmt = (s: number) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toFixed(1).padStart(4, '0')}`

export function WavePlayer({ audioUrl, seekTo, onTimeUpdate }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#262f45',
      progressColor: '#06b6d4',
      cursorColor: '#22d3ee',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      height: 70,
      normalize: true,
    })
    ws.load(audioUrl)
    ws.on('ready', () => setDuration(ws.getDuration()))
    ws.on('timeupdate', (t) => {
      setCurrent(t)
      onTimeUpdate?.(t)
    })
    ws.on('play', () => setPlaying(true))
    ws.on('pause', () => setPlaying(false))
    ws.on('finish', () => setPlaying(false))
    wsRef.current = ws
    return () => {
      ws.destroy()
      wsRef.current = null
      setPlaying(false)
    }
  }, [audioUrl])

  useEffect(() => {
    if (seekTo && wsRef.current && duration > 0) {
      wsRef.current.setTime(Math.min(seekTo.time, duration))
      wsRef.current.play().catch(() => {})
    }
  }, [seekTo])

  return (
    <div className="wave-player">
      <div className="wave-controls">
        <button className="primary-btn" onClick={() => wsRef.current?.playPause()}>
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {playing ? 'Pause' : 'Play'}
        </button>
        <span className="wave-time">
          <Volume2 size={14} /> {fmt(current)} / {fmt(duration)}
        </span>
      </div>
      <div ref={containerRef} className="wavesurfer-box" />
    </div>
  )
}
