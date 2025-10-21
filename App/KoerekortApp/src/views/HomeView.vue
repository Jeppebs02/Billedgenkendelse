<template>
  <div class="grid" style="gap:16px">
    <!-- Upload -->
    <UploadCard @submit="handleSubmit" />

    <!-- Resultat-header -->
    <section v-if="data" class="card" style="padding:16px">
      <div class="row" style="justify-content:space-between;align-items:center">
        <div>
          <h2 style="margin:0">Opsummering</h2>
          <p class="text-muted" style="margin:.25rem 0 0">
            {{ data.passed ? 'Billedet opfylder kravene.' : 'Billedet opfylder ikke alle krav.' }}
          </p>
        </div>

        <div class="row" style="gap:8px; align-items:center">
          <span
            class="badge"
            :class="{
              'text-good': data.decision === 'APPROVED',
              'text-bad': data.decision === 'REJECTED'
            }"
          >
            {{ data.decision }}
          </span>
          <button class="btn btn-primary" @click="reset">Prøv igen</button>
        </div>
      </div>

      <div class="hr"></div>

      <!-- Kort “human” summary hvis backend sender det -->
      <div v-if="data.summary" class="row" style="gap:8px; flex-wrap:wrap">
        <span class="badge" :class="badgeClass(data.summary.face)">Ansigt</span>
        <span class="badge" :class="badgeClass(data.summary.single_face)">Enkelt ansigt</span>
        <span class="badge" :class="badgeClass(data.summary.mouth_closed)">Mund lukket</span>
        <span class="badge" :class="badgeClass(data.summary.no_hat)">Ingen hat</span>
        <span class="badge" :class="badgeClass(data.summary.no_glasses)">Ingen briller</span>
      </div>
    </section>

    <!-- Checks-liste -->
    <section v-if="data" class="card" style="padding:16px">
      <div class="row" style="justify-content:space-between;align-items:center">
        <h3 style="margin:0">Detaljerede checks</h3>
        <span class="badge">Threshold: {{ data.threshold }}</span>
      </div>
      <div class="hr"></div>

      <div class="list">
        <article v-for="c in data.checks" :key="c.key" class="list-item">
          <span
            aria-hidden="true"
            :class="{
              'text-good': c.status==='pass',
              'text-warn': c.status==='warn',
              'text-bad' : c.status==='fail'
            }"
            style="font-size:20px; line-height:1"
          >●</span>

          <div style="flex:1">
            <div class="row" style="gap:8px; align-items:center">
              <strong>{{ c.label }}</strong>
              <small class="badge" v-if="c.severity">{{ c.severity }}</small>
            </div>
            <p class="text-muted" style="margin:.25rem 0 0">
              {{ c.message || fallbackMessage(c.status) }}
            </p>

            <details v-if="c.details" style="margin-top:.5rem">
              <summary class="text-muted">Detaljer</summary>
              <pre style="white-space:pre-wrap; margin:.5rem 0 0; font-size:.85rem; opacity:.9">
{{ prettyJson(c.details) }}</pre>
            </details>
          </div>
        </article>
      </div>
    </section>

    <!-- Metadata (valgfri hvis du senere sender disse felter) -->
    <section v-if="data" class="card" style="padding:16px">
      <h3 style="margin-top:0">Metadata</h3>
      <div class="kv">
        <div>Beslutning</div><div>{{ data.decision }}</div>
        <div>Threshold</div><div>{{ data.threshold }}</div>
        <div v-if="data.meta?.width">Mål</div>
        <div v-if="data.meta?.width">{{ data.meta?.width }} × {{ data.meta?.height }} px</div>
        <div v-if="data.meta?.mime">Format</div>
        <div v-if="data.meta?.mime">{{ data.meta?.mime }}</div>
        <div v-if="data.meta?.size_bytes">Filstørrelse</div>
        <div v-if="data.meta?.size_bytes">{{ prettyBytes(data.meta?.size_bytes) }}</div>
      </div>
    </section>

    <!-- Forhåndsinfo -->
    <section v-if="!data" class="card" style="padding:16px">
      <h3 style="margin-top:0">Krav (kort fortalt)</h3>
      <ul class="text-muted" style="margin:0 0 0 1rem;line-height:1.6">
        <li>Ensartet lys baggrund</li>
        <li>Ansigt tydeligt, neutralt udtryk</li>
        <li>Ingen hat/solbriller (medmindre tilladt)</li>
        <li>Øjne åbne og synlige</li>
        <li>God skarphed og korrekt beskæring</li>
      </ul>
    </section>

    <!-- Fejl -->
    <p v-if="err" class="text-bad">{{ err }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import UploadCard from '../components/UploadCard.vue'
import { analyzePhoto } from '../services/api'

const data = ref(null)
const err = ref('')

// Label-helpers
function humanizeRequirement(req) {
  if (!req) return ''
  const s = String(req).split('.').pop()
  return s.replace(/_/g, ' ').toLowerCase().replace(/(^|\s)\S/g, m => m.toUpperCase())
}
function toStatus(passed) { return passed ? 'pass' : 'fail' }
function fallbackMessage(status){
  if(status==='pass') return 'Opfylder kravet.'
  if(status==='warn') return 'Næsten – overvej at tage et nyt billede.'
  return 'Opfylder ikke kravet.'
}
function prettyJson(obj){
  try { return JSON.stringify(obj, null, 2) } catch { return String(obj) }
}
function prettyBytes(n) {
  if (!n && n !== 0) return ''
  const units = ['B','KB','MB','GB']
  let i = 0; let v = n
  while (v >= 1024 && i < units.length-1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}
function badgeClass(passed){
  return passed ? 'text-good' : 'text-bad'
}

// Submit → kald Flask /analyze
async function handleSubmit({ file, onProgress }){
  err.value = ''
  data.value = null

  // Simpel progress-animation (kan erstattes med XHR onprogress)
  let t = 15
  const timer = setInterval(()=> { t = Math.min(95, t + 7); onProgress?.(t) }, 200)

  try{
    const { status, data: api } = await analyzePhoto({ file, threshold: 0.5 })

    const checksUi = (api.checks ?? []).map(c => ({
      key: String(c.requirement),
      label: humanizeRequirement(c.requirement),
      status: toStatus(c.passed),
      message: c.message || '',
      severity: c.severity || '',
      details: c.details || null,
    }))

    data.value = {
      decision: api.decision,            // "APPROVED" | "REJECTED"
      passed: Boolean(api.passed),
      threshold: api.threshold,
      summary: api.summary ?? {},

      // UI-felter
      score: api.passed ? 1 : 0,
      checks: checksUi,

      // Valgfri meta, udfyld hvis/naar backend sender disse
      meta: {
        width: api.image_width ?? undefined,
        height: api.image_height ?? undefined,
        mime: api.mime ?? undefined,
        size_bytes: api.size_bytes ?? undefined,
      }
    }

    // status 200 = APPROVED, 422 = REJECTED (gyldigt svar)
    if (!api || (status !== 200 && status !== 422)) {
      throw new Error('Uventet svar fra serveren.')
    }

    onProgress?.(100)
  }catch(e){
    err.value = e?.message || 'Kunne ikke analysere billedet.'
  }finally{
    clearInterval(timer)
    setTimeout(()=> onProgress?.(0), 400)
  }
}

function reset(){
  data.value = null
  err.value = ''
}
</script>
