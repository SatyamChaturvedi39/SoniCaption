export type Category = 'speech' | 'sound_effect' | 'background_noise' | 'silence'
export type Status = 'ok' | 'alert' | 'warn' | 'idle'

export interface TimelineItem {
  id: string
  start: number
  end: number
  category: Category
  label: string
  text: string
  speaker: string | null
  speaker_id: number | null
  confidence: number | null
  status: Status
  key?: string
  delivery_style?: string
}

export interface Task {
  id: string
  task: string
  owner: string
  owner_id: number | null
  owner_name: string | null
  due: string | null
  priority: 'low' | 'medium' | 'high'
  kind: string
  confidence: number
  source_text: string
  start: number
  said_by: string
  requested_by: string | null
  confirmed?: boolean
}

export interface SpeakerProfile {
  speaker_id: number
  speaker: string
  label: string
  name: string | null
  talk_seconds: number
  share: number
  turns: number
  words: number
  words_per_minute: number
  topics: string[]
  key_points: { t: number; text: string; kind: string }[]
  questions: string[]
  tasks: string[]
  summary: string
  llm_summary?: string
}

export interface Scenario {
  label: string
  confidence: number
  evidence: string[]
  alternatives: { label: string; score: number }[]
  speakers: number
  llm_label?: string
}

export interface Nlu {
  scenario: Scenario
  speakers: SpeakerProfile[]
  tasks: Task[]
  decisions: { t: number; speaker: string; text: string }[]
  questions: { speaker: string; text: string }[]
  provider: string
  warnings: string[]
}

export interface SpeakerPattern {
  speaker_id: number
  speaker: string
  seconds_analysed: number
  mean_f0_hz: number
  f0_std_hz: number
  voiced_ratio: number
  mfcc_mean: number[]
  mfcc_std: number[]
}

export interface HmmModel {
  states: string[]
  symbols: string[]
  pi: number[]
  A: number[][]
  B: number[][]
}

export interface Analysis {
  warnings?: string[]
  filename: string
  duration: number
  denoised: boolean
  vad_speech_ratio: number
  session_id: number | null
  asr: { status: 'ok' | 'unavailable' | 'skipped'; provider: string | null; language: string | null; warnings: string[]; error: string | null }
  scene: { segments: SceneSegment[]; stats: Record<string, { seconds: number; share: number }> }
  diarization: { n_speakers: number; backend: string; turns: { start: number; end: number; speaker_id: number }[]; projection: { x: number; y: number; speaker_id: number; t: number }[]; threshold: number | null }
  speakers: { speaker_id: number; label: string; name: string | null; seconds: number; share: number | null; topics: string[]; summary: string; turns: number }[]
  utterances: { start: number; end: number; text: string; speaker_id: number; speaker: string; delivery_style?: string }[]
  timeline: TimelineItem[]
  nlu: Nlu
  patterns: {
    spectral: {
      covered_seconds: number
      waveform?: number[]
      waveform_denoised?: number[]
      rms?: number[]
      stft_db?: number[][]
      stft_freq_max_hz?: number
      mfcc?: number[][]
      f0_contour_hz?: number[]
      f0_mean_hz?: number
      voiced_ratio?: number
      lpc_formants_hz?: number[]
      lpc_envelope_db?: number[]
      snr_before_db?: number | null
      snr_after_db?: number | null
      snr_gain_db?: number | null
    }
    speakers: SpeakerPattern[]
    hmm: {
      scene: { hop_s: number; observations: string[]; path: string[]; log_likelihood: number | null; best_path_log_prob: number | null; model: HmmModel | null }
      prosody: { step_s: number; observations: string[]; best_path: string[]; log_likelihood: number } | null
    }
  }
  timings_ms: Record<string, number>
}

export interface SceneSegment {
  id: string
  state: Category
  label: string
  key: string
  start: number
  end: number
  confidence: number
  speaker_id?: number
}

export interface Health {
  status: string
  asr: { available: boolean; provider: string; hint: string | null }
  ffmpeg_available: boolean
  speaker_encoder: string
  groq_configured: boolean
}

export type LiveEvent =
  | { type: 'ready'; asr: Health['asr'] }
  | { type: 'status'; asr: string; message: string }
  | { type: 'segment'; id: string; category: Category; label: string; key?: string; start: number; end: number; confidence: number; status: Status; open: boolean; speaker?: string; speaker_id?: number }
  | { type: 'transcript'; id: string; start: number; end: number; text: string; speaker: string; speaker_id: number }
  | { type: 'insights'; scenario: Scenario; tasks: Task[]; speakers: SpeakerProfile[]; decisions: Nlu['decisions'] }
  | { type: 'progress'; stage: string; progress: number; message: string }
  | { type: 'final'; result: Analysis }
  | { type: 'error'; message: string }
