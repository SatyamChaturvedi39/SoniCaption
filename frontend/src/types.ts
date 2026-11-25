export type TimelineItem = {
  id: string
  start: number
  end: number
  category: 'speech' | 'sound_event' | string
  label: string
  speaker?: string
  display_text: string
  confidence: number | null
  source?: string
  overlaps_speech?: boolean
}

export type ReminderItem = {
  id: number
  session_id?: number | null
  title: string
  due_time: string
  priority: 'low' | 'medium' | 'high' | string
  trigger_source: 'speech_action' | 'acoustic_alert' | 'manual' | string
  status: 'pending' | 'completed'
  created_at?: string
}

export type AlertItem = {
  id: number
  created_at: string
  event_label: string
  confidence: number
  severity: 'high' | 'medium' | 'info' | string
  snippet_text: string
}

export type Analysis = {
  session_id?: number | null
  filename: string
  duration: number
  sample_rate: number
  mode: string
  summary?: string
  reminders?: ReminderItem[]
  speech_segments: any[]
  sound_events: any[]
  timeline: TimelineItem[]
  metrics: Record<string, any>
}

export type SessionLog = {
  id: number
  created_at: string
  filename: string
  source_type: string
  duration: number
  speech_source: string
  vad_speech_ratio: number
  snr_before_db: number
  snr_after_db: number
  denoised: boolean
  summary: string
  agent_reply?: string
  timeline: TimelineItem[]
  metrics: Record<string, any>
}

export type AgentTurnResponse = {
  session_id: number
  transcript: string
  sound_events: any[]
  pitch_stats: Record<string, any>
  reply: string
  provider: string
  new_reminders: ReminderItem[]
  warnings: string[]
}
