
<script setup>
import { ref } from 'vue'
import { Camera, CameraResultType } from '@capacitor/camera'
import { IonButton, useIonRouter } from '@ionic/vue';
import { useDataStore } from '@/stores/dataStore.js'




// Preview
const imageSrc = ref(null)
// Bytes to be uploaded
const lastBlob = ref(null)
// Threshhold is optional and defaults to 0.5
const threshold = ref('0.5')
const loading = ref(false)
const serverResponse = ref(null)
const httpMsg = ref('')
const dataStore = useDataStore()
const ionrouter = useIonRouter()

async function takePicture() {
  httpMsg.value = ''
  serverResponse.value = null

  const photo = await Camera.getPhoto({
    quality: 90,
    allowEditing: false,
    resultType: CameraResultType.Uri, // returns webPath/URL
  })

  imageSrc.value = photo.webPath || null
  dataStore.setImageSrc(photo.webPath)

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
    const resp = await fetch('https://api.terragrouplabs.net/analyze', {
      method: 'POST',
      body: form,
    })

    const data = await resp.json()
    serverResponse.value = data

    dataStore.setJsonResult(serverResponse.value)

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
     ionrouter.navigate('/result', 'forward', 'replace');

  }
}
</script>

<template>
  <div class="responsive-grid">
    <div class="card">
      <!-- image -->
      <div class="photo-frame" :class="{ 'has-image': !!imageSrc }">
        <img v-if="imageSrc" :src="imageSrc" alt="Uploaded photo" class="photo-img" />
        <div v-else class="photo-placeholder">
          <ion-icon name="image-outline"></ion-icon>
          <p>Upload a photo</p>
          <small>Portrait, face centered, plain background</small>
        </div>
      </div>


      <!-- The “form” UI -->
      <form @submit.prevent="addPhoto" enctype="multipart/form-data" style="display:grid; gap: 8px;">
        <!-- Upload Picture button -->
        <ion-button class="cool-btn" @click="takePicture">Upload photo</ion-button>

        <!-- Analyse button -->
        <ion-button class="cool-btn" type="submit" :disabled="loading">
          <!-- if loading is true, show Analyzing, if false show Analyze photo -->
          {{ loading ? 'Analyzing...' : 'Analyze photo' }}
        </ion-button>
      </form>



      <!-- Server Feedback -->
      <p v-if="httpMsg" style="color:#b00020;">{{ httpMsg }}</p>

      <details v-if="serverResponse">
        <summary>Server response</summary>
        <pre style="white-space: pre-wrap;">{{ JSON.stringify(serverResponse, null, 2) }}</pre>
      </details>
    </div>
  </div>
</template>


