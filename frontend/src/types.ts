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
  delivery_style?: string
  mean_f0_hz?: number
  energy_ratio?: number
  speaking_rate_wps?: number
}

export type DtwSpotMatch = {
  start: number
  end: number
  normalized_distance: number
  similarity_pct: number
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

export type SimilarSession = {
  session_id: number
  filename: string
  created_at: string
  normalized_distance: number
  similarity_pct: number
  summary: string
}

export type Analysis = {
  session_id?: number | null
  filename: string
  duration: number
  sample_rate: number
  mode: string
  summary?: string
  reminders?: ReminderItem[]
  similar_sessions?: SimilarSession[]
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
  new_notes?: ReminderItem[]
  matched_sessions?: { session_id: number; filename: string; summary: string }[]
  warnings: string[]
}
