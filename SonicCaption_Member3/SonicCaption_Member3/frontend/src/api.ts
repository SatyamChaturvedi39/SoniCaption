import type { Analysis } from './types'
const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
export async function analyze(file: File): Promise<Analysis> {
  const form = new FormData(); form.append('file', file)
  const r = await fetch(`${API}/api/v1/audio/analyze`, {method:'POST', body:form})
  if (!r.ok) { const d = await r.json().catch(()=>({})); throw new Error(d.detail || 'Analysis failed') }
  return r.json()
}
export async function getLab(path:string) {
  const r = await fetch(`${API}${path}`, {method:'POST'})
  if (!r.ok) throw new Error('Could not load demo data')
  return r.json()
}
export async function exportResult(fmt:string, data:any) {
  const r = await fetch(`${API}/api/v1/export/${fmt}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)})
  if (!r.ok) throw new Error('Export failed')
  const blob = await r.blob(), url = URL.createObjectURL(blob), a = document.createElement('a')
  a.href=url; a.download=`soniccaption.${fmt}`; a.click(); URL.revokeObjectURL(url)
}
