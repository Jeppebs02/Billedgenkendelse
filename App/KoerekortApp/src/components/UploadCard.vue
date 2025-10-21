<template>
  <section class="card shadow" role="region" aria-label="Upload billede">
    <div class="grid" style="padding:20px">
      <div
        class="dropzone"
        :class="{ dragover: isDragOver }"
        @dragover.prevent="isDragOver = true"
        @dragleave.prevent="isDragOver = false"
        @drop.prevent="onDrop"
      >
        <p class="text-muted">Træk & slip billede her – eller</p>
        <div class="row" style="justify-content:center">
          <label class="btn btn-primary" :for="inputId">Vælg billede</label>
          <button class="btn btn-ghost" @click="openCamera" type="button">Brug kamera</button>
        </div>
        <input
          :id="inputId"
          ref="fileInput"
          type="file"
          accept="image/*"
          class="sr-only"
          @change="onFileChange"
        />
        <!-- skjult capture input til kamera -->
        <input
          ref="cameraInput"
          type="file"
          accept="image/*"
          capture="environment"
          class="sr-only"
          @change="onFileChange"
        />
      </div>

      <div v-if="file" class="preview">
        <img :src="previewUrl" alt="Forhåndsvisning af billede" />
        <div class="grid" style="gap:6px">
          <div class="kv">
            <div>Filnavn</div><div>{{ file.name }}</div>
            <div>Størrelse</div><div>{{ prettyBytes(file.size) }}</div>
            <div>MIME</div><div>{{ file.type || 'ukendt' }}</div>
          </div>
          <div class="row" style="gap:8px">
            <button class="btn" @click="clear">Skift billede</button>
            <button class="btn btn-primary" :disabled="loading" @click="submit">
              {{ loading ? 'Validerer…' : 'Tjek billede' }}
            </button>
          </div>
          <div v-if="loading" class="progress" aria-label="Uploader og validerer">
            <div :style="{ width: progress + '%' }"></div>
          </div>
          <p v-if="error" class="text-bad" style="margin:0">{{ error }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, computed } from 'vue'

const emit = defineEmits(['submit'])

const file = ref(null)
const previewUrl = ref('')
const inputId = `file-${Math.random().toString(36).slice(2)}`
const fileInput = ref(null)
const cameraInput = ref(null)

const loading = ref(false)
const error = ref('')
const isDragOver = ref(false)
const progress = ref(0)

function prettyBytes(n) {
  if (!n && n !== 0) return ''
  const units = ['B','KB','MB','GB']
  let i = 0; let v = n
  while (v >= 1024 && i < units.length-1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

function onDrop(e){
  isDragOver.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) setFile(f)
}

function onFileChange(e){
  const f = e.target.files?.[0]
  if (f) setFile(f)
}

function setFile(f){
  error.value = ''
  file.value = f
  previewUrl.value = URL.createObjectURL(f)
  progress.value = 0
}

function clear(){
  error.value = ''
  file.value = null
  previewUrl.value = ''
  progress.value = 0
  fileInput.value && (fileInput.value.value = '')
  cameraInput.value && (cameraInput.value.value = '')
}

function openCamera(){
  cameraInput.value?.click()
}

async function submit(){
  if (!file.value) return
  loading.value = true
  error.value = ''
  progress.value = 15 // lille UI feedback

  try{
    // vi emitter filen – selve API-kaldet håndteres i parent (HomeView)
    await emit('submit', { file: file.value, onProgress: (p)=> (progress.value = p) })
    progress.value = 100
  }catch(err){
    error.value = err?.message || 'Noget gik galt.'
  }finally{
    loading.value = false
    setTimeout(()=> (progress.value = 0), 800)
  }
}
</script>
