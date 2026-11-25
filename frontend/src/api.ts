import type { AgentTurnResponse, AlertItem, Analysis, ReminderItem, SessionLog } from './types'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function reqJson<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${url}`, init)
  if (!r.ok) {
    const d = await r.json().catch(() => ({}))
    throw new Error(d.detail || `Request failed (${r.status})`)
  }
  return r.json()
}

export const getHealth = () => reqJson<any>('/api/health')

export const setGroqKey = (apiKey: string) =>
  reqJson<{ groq_configured: boolean }>('/api/v1/settings/groq-key', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: apiKey }),
  })

export async function analyzeAudio(
  file: File,
  opts: { denoise: boolean; sourceType?: string } = { denoise: true }
): Promise<Analysis> {
  const form = new FormData()
  form.append('file', file)
  form.append('denoise', String(opts.denoise))
  form.append('source_type', opts.sourceType || 'upload')
  form.append('save_log', 'true')
  return reqJson<Analysis>('/api/v1/audio/analyze', { method: 'POST', body: form })
}

export async function sendAgentTurn(params: {
  file?: File | null
  text?: string
  denoise?: boolean
  history?: { role: string; text: string }[]
}): Promise<AgentTurnResponse> {
  const form = new FormData()
  if (params.file) form.append('file', params.file)
  if (params.text) form.append('text', params.text)
  form.append('denoise', String(params.denoise ?? true))
  form.append('history_json', JSON.stringify(params.history || []))
  return reqJson<AgentTurnResponse>('/api/v1/agent/turn', { method: 'POST', body: form })
}

export async function sendWatchChunk(file: File, denoise = true) {
  const form = new FormData()
  form.append('file', file)
  form.append('denoise', String(denoise))
  return reqJson<any>('/api/v1/watch/chunk', { method: 'POST', body: form })
}

export const getSessions = (search = '') =>
  reqJson<{ sessions: SessionLog[] }>(`/api/v1/sessions?search=${encodeURIComponent(search)}`)

export const deleteSession = (id: number) =>
  reqJson<{ deleted: boolean }>(`/api/v1/sessions/${id}`, { method: 'DELETE' })

export const getReminders = () => reqJson<{ reminders: ReminderItem[] }>('/api/v1/reminders')

export const createReminder = (payload: { title: string; due_time?: string; priority?: string }) =>
  reqJson<ReminderItem>('/api/v1/reminders', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

export const toggleReminder = (id: number, status: 'pending' | 'completed') =>
  reqJson<ReminderItem>(`/api/v1/reminders/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  })

export const deleteReminder = (id: number) =>
  reqJson<{ deleted: boolean }>(`/api/v1/reminders/${id}`, { method: 'DELETE' })

export const getAlerts = () => reqJson<{ alerts: AlertItem[] }>('/api/v1/alerts')

export async function getSignalLab(file?: File | null, useLast = true) {
  if (file) {
    const form = new FormData()
    form.append('file', file)
    return reqJson<any>(`/api/v1/labs/signal?use_last=${useLast}`, { method: 'POST', body: form })
  }
  return reqJson<any>(`/api/v1/labs/signal?use_last=${useLast}`, { method: 'POST' })
}

export async function getDtwLab(fileA?: File | null, fileB?: File | null) {
  if (fileA && fileB) {
    const form = new FormData()
    form.append('file_a', fileA)
    form.append('file_b', fileB)
    return reqJson<any>('/api/v1/labs/dtw', { method: 'POST', body: form })
  }
  return reqJson<any>('/api/v1/labs/dtw', { method: 'POST' })
}

export const getHmmLab = (preset: string, observations?: string[]) =>
  reqJson<any>('/api/v1/labs/hmm/viterbi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset, ...(observations?.length ? { observations } : {}) }),
  })

export async function exportResult(fmt: string, data: any) {
  const r = await fetch(`${API}/api/v1/export/${fmt}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!r.ok) throw new Error('Export failed')
  const blob = await r.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `soniccaption.${fmt}`
  a.click()
  URL.revokeObjectURL(url)
}
