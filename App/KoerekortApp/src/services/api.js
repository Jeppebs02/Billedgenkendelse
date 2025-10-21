// Tip: Sæt endpoint via .env: VITE_API_BASE
const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export async function validatePhoto({ file }) {
  const form = new FormData()
  form.append('file', file) // backend key: "file"

  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `API error ${res.status}`)
  }
  /** Forventet respons (justér til din API):
   * {
   *   id: "req_123",
   *   status: "ok" | "fail" | "warn",
   *   score: 0.0..1.0,
   *   summary: "Kort opsummering",
   *   checks: [
   *     { key:"background_uniform", label:"Ensartet baggrund", status:"pass|fail|warn", message:"..." },
   *     { key:"face_visible", label:"Ansigt tydeligt", status:"pass", message:"" },
   *     ...
   *   ],
   *   meta: { width:1024, height:1024, mime:"image/jpeg", size_bytes: 123456 },
   *   report_url: "https://.../report.pdf" // valgfrit
   * }
   */
  return await res.json()
}
