<template>
  <div class="grid" style="gap:16px">
    <UploadCard @submit="handleSubmit" />

    <ResultHeader
      v-if="data"
      :summary="data.summary"
      :reportUrl="data.report_url"
      @retry="reset"
    />

    <ChecksList
      v-if="data"
      :checks="data.checks"
      :score="data.score"
    />

    <section v-if="data" class="card" style="padding:16px">
      <h3 style="margin-top:0">Metadata</h3>
      <div class="kv">
        <div>Mål</div><div>{{ data.meta?.width }} × {{ data.meta?.height }} px</div>
        <div>Format</div><div>{{ data.meta?.mime }}</div>
        <div>Filstørrelse</div><div>{{ prettyBytes(data.meta?.size_bytes) }}</div>
        <div>Request ID</div><div class="text-muted">{{ data.id }}</div>
      </div>
    </section>

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

    <p v-if="err" class="text-bad">{{ err }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import UploadCard from '../components/UploadCard.vue'
import ChecksList from '../components/ChecksList.vue'
import ResultHeader from '../components/ResultHeader.vue'
import { validatePhoto } from '../services/api'

const data = ref(null)
const err = ref('')

function prettyBytes(n) {
  if (!n && n !== 0) return ''
  const units = ['B','KB','MB','GB']
  let i = 0; let v = n
  while (v >= 1024 && i < units.length-1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

async function handleSubmit({ file, onProgress }){
  err.value = ''
  data.value = null

  // Simpel progress simulation (du kan udskifte med real upload progress via XHR)
  let t = 15
  const timer = setInterval(()=> {
    t = Math.min(95, t + 7)
    onProgress?.(t)
  }, 200)

  try{
    const res = await validatePhoto({ file })
    data.value = res
    onProgress?.(100)
  }catch(e){
    err.value = e?.message || 'Kunne ikke validere billedet.'
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
