<script setup>
import { ref } from 'vue'
import { IonButton, useIonRouter } from '@ionic/vue';
import { useDataStore } from '@/stores/dataStore.js'

const httpMsg = ref('')
const dataStore = useDataStore()
const imageSrc = ref(null)

imageSrc.value = dataStore.getImageSrc()
</script>

<template>
  <div style="display:grid; gap:12px; max-width: 420px;">

    <img
      v-if="imageSrc"
      :src="imageSrc"
      alt="snapshot"
      style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;"
    />

  </div>
</template>

<style scoped>

</style>
