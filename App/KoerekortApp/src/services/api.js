// src/services/api.js
const BASE = import.meta.env.VITE_API_BASE ?? 'http://192.168.1.89:5001'

export async function analyzePhoto({ file, threshold = 0.5 }) {
  const form = new FormData()
  form.append('file', file)
  form.append('threshold', String(threshold))

  const res = await fetch(`${BASE}/analyze`, { method: 'POST', body: form })

  // prøv altid at parse JSON – også ved 422
  let data
  try { data = await res.json() }
  catch {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Uventet svar (${res.status})`)
  }

  // 200 = APPROVED, 422 = REJECTED (gyldigt svar)
  if (!res.ok && res.status !== 422) {
    throw new Error(data?.error || data?.message || `API-fejl (${res.status})`)
  }
  return { status: res.status, data }
}
