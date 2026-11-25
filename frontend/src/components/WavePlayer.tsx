import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Pause, Play, Volume2, Sparkles, AlertTriangle } from 'lucide-react'
import type { TimelineItem } from '../types'

type Props = {
  audioUrl: string
  seekTo?: { time: number; seq: number } | null
  timeline?: TimelineItem[]
  onTimeUpdate?: (time: number) => void
  onSeek?: (time: number) => void
}

const fmt = (s: number) =>
  `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toFixed(1).padStart(4, '0')}`

export function WavePlayer({ audioUrl, seekTo, timeline, onTimeUpdate, onSeek }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#252538',
      progressColor: '#06b6d4',
      cursorColor: '#f43f5e',
      cursorWidth: 2,
      barWidth: 2.5,
      barGap: 2,
      barRadius: 2,
      height: 72,
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

  const handleLaneSeek = (t: number) => {
    if (wsRef.current && duration > 0) {
      wsRef.current.setTime(Math.min(t, duration))
      wsRef.current.play().catch(() => {})
    }
    onSeek?.(t)
  }

  const speechItems = timeline?.filter((item) => item.category === 'speech') || []
  const eventItems = timeline?.filter((item) => item.category === 'sound_event') || []
  const emphasisItems = speechItems.filter(
    (item) => item.delivery_style && item.delivery_style.toLowerCase().includes('emphasis')
  )

  return (
    <div className="wave-player">
      <div className="wave-controls">
        <button className="primary-btn" onClick={() => wsRef.current?.playPause()}>
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {playing ? 'Pause' : 'Play Audio'}
        </button>
        <span className="wave-time">
          <Volume2 size={14} /> {fmt(current)} / {fmt(duration)}
        </span>
        {emphasisItems.length > 0 && (
          <span className="tag purple" style={{ marginLeft: 'auto' }}>
            <Sparkles size={12} /> {emphasisItems.length} Emphasis Keypoints
          </span>
        )}
        {eventItems.length > 0 && (
          <span className="tag warn">
            <AlertTriangle size={12} /> {eventItems.length} Sound Events
          </span>
        )}
      </div>

      <div ref={containerRef} className="wavesurfer-box" />

      {duration > 0 && timeline && timeline.length > 0 && (
        <div className="multi-lane-scrubber">
          {/* Speech Track */}
          <div className="lane-row">
            <span className="lane-title">Speech / VQ</span>
            <div className="lane-track">
              {speechItems.map((item) => {
                const left = (item.start / duration) * 100
                const width = Math.max(1.5, ((item.end - item.start) / duration) * 100)
                return (
                  <div
                    key={item.id}
                    className="lane-block speech"
                    style={{ left: `${left}%`, width: `${width}%` }}
                    title={`${item.speaker || 'Speaker'}: ${item.display_text} (${item.start.toFixed(1)}s)`}
                    onClick={() => handleLaneSeek(item.start)}
                  >
                    {item.speaker || 'Spk'}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Emphasis Track */}
          {emphasisItems.length > 0 && (
            <div className="lane-row">
              <span className="lane-title">Vocal Stress</span>
              <div className="lane-track">
                {emphasisItems.map((item) => {
                  const left = (item.start / duration) * 100
                  const width = Math.max(2, ((item.end - item.start) / duration) * 100)
                  return (
                    <div
                      key={`emp-${item.id}`}
                      className="lane-block emphasis"
                      style={{ left: `${left}%`, width: `${width}%` }}
                      title={`Key Emphasis: ${item.display_text} (${item.start.toFixed(1)}s)`}
                      onClick={() => handleLaneSeek(item.start)}
                    >
                      ★ {item.mean_f0_hz ? `${item.mean_f0_hz}Hz` : 'Stress'}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Sound Events Track */}
          {eventItems.length > 0 && (
            <div className="lane-row">
              <span className="lane-title">Acoustic ESC</span>
              <div className="lane-track">
                {eventItems.map((item) => {
                  const left = (item.start / duration) * 100
                  const width = Math.max(2, ((item.end - item.start) / duration) * 100)
                  return (
                    <div
                      key={item.id}
                      className="lane-block event"
                      style={{ left: `${left}%`, width: `${width}%` }}
                      title={`${item.label} (${Math.round((item.confidence || 0) * 100)}%)`}
                      onClick={() => handleLaneSeek(item.start)}
                    >
                      {item.label}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
