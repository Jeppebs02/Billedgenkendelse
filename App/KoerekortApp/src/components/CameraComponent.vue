
<script setup>
import { ref } from 'vue'
import { Camera, CameraResultType } from '@capacitor/camera'
import { IonButton } from '@ionic/vue';


// Preview
const imageSrc = ref(null)
// Bytes to be uploaded
const lastBlob = ref(null)
// Threshhold is optional and defaults to 0.5
const threshold = ref('0.5')
const loading = ref(false)
const serverResponse = ref(null)
const httpMsg = ref('')

async function takePicture() {
  httpMsg.value = ''
  serverResponse.value = null

  const photo = await Camera.getPhoto({
    quality: 90,
    allowEditing: false,
    resultType: CameraResultType.Uri, // returns webPath/URL
  })

  imageSrc.value = photo.webPath || null

  // Convert the captured image URL to a Blob so it can be uploaded :)
  if (photo.webPath) {
    const res = await fetch(photo.webPath)
    lastBlob.value = await res.blob()
  } else {
    lastBlob.value = null
  }
}

async function addPhoto() {
  httpMsg.value = ''
  serverResponse.value = null

  if (!lastBlob.value) {
    httpMsg.value = 'Please take a photo or choose a file first.'
    return
  }

  const form = new FormData()
  // Backend expects the field name 'file', look in application.py under the /analyze route.
  form.append('file', lastBlob.value, 'photo.jpg')
  form.append('threshold', threshold.value) // optional

  try {
    loading.value = true
    // DO NOT set Content-Type manually. The browser sets it for FormData.
    const resp = await fetch('http://127.0.0.1:5001/analyze', {
      method: 'POST',
      body: form,
    })

    const data = await resp.json()
    serverResponse.value = data

    if (!resp.ok) {
      // 422 "Unprocessable Entity" is rejected
      httpMsg.value = data?.error || `Server returned status ${resp.status}`
    }
    if (resp.ok) {
      httpMsg.value = data?.error || `Server returned status ${resp.status}`
    }
  } catch (err) {
    httpMsg.value = String(err)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div style="display:grid; gap:12px; max-width: 420px;">

    <img
      v-if="imageSrc"
      :src="imageSrc"
      alt="snapshot"
      style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;"
    />

    <!-- The “form” UI -->
    <form @submit.prevent="addPhoto" enctype="multipart/form-data" style="display:grid; gap: 8px;">
      <ion-button class="cool-btn" @click="takePicture">Upload photo</ion-button>
      <ion-button class="cool-btn" type="submit" :disabled="loading">
        <!-- if loading is true, show Analyzing, if false show Analyze photo -->
        {{ loading ? 'Analyzing...' : 'Analyze photo' }}
      </ion-button>
    </form>

<style scoped>
/* === Layout centreret === */
.page-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 80px);
  background: #fff;
}

.content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
  width: 100%;
  max-width: 360px;
  padding: 20px;
  box-sizing: border-box;
}

/* === Billede med fast størrelse === */
.preview {
  width: 200px;
  height: 200px;
  object-fit: cover;
  border-radius: 12px;
  border: 2px solid #ddd;
  box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

/* === Knapper === */
.button-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
}

.cool-btn {
  --background: linear-gradient(135deg, #000 0%, #fbf6f6 60%);
  --color: black;
  --border-radius: 12px;
  --padding-top: 14px;
  --padding-bottom: 14px;
  --padding-start: 20px;
  --padding-end: 20px;
  --ripple-color: rgba(255, 255, 255, 0.5);
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  transition: transform 0.15s ease, box-shadow 0.3s ease;
}




/* === Feedback === */
.error {
  color: #d20000;
  text-align: center;
  font-size: 14px;
  margin-top: 10px;
}

.response {
  width: 100%;
  word-wrap: break-word;
  background: #fafafa;
  border-radius: 8px;
  padding: 10px;
  font-size: 13px;
}
</style>
