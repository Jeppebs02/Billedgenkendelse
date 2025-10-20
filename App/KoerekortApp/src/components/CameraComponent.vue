<script setup>
import { ref } from 'vue'
import { Camera, CameraResultType } from '@capacitor/camera'

const imageSrc = ref(null)

async function takePicture() {
  const photo = await Camera.getPhoto({
    quality: 90,
    allowEditing: true,
    resultType: CameraResultType.Uri,
  })
  imageSrc.value = photo.webPath || null
}
</script>

<template>
  <div style="display:grid; gap:12px;">
    <button @click="takePicture">Take photo</button>
    <img v-if="imageSrc" :src="imageSrc" alt="snapshot" style="max-width: 240px;" />
  </div>
</template>

<style scoped>
/* optional */
</style>
