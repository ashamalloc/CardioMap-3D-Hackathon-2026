const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  const payload = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(payload.detail?.message || payload.detail || `Request failed: ${res.status}`)
  return payload
}

export const getStatus = () => request('/api/status')
export const getSchema = () => request('/api/schema')
export const getMetrics = () => request('/api/metrics')
export const getImportance = () => request('/api/importance')
export const predict = (features) => request('/api/predict', { method: 'POST', body: JSON.stringify({ features }) })
