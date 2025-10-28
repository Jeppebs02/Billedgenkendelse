<script setup>
import { computed, ref } from 'vue'
import { IonButton, useIonRouter } from '@ionic/vue'
import { useDataStore } from '@/stores/dataStore.js'


const httpMsg = ref('')
const dataStore = useDataStore()
const ionRouter = useIonRouter()
const imageSrc = computed(()=>dataStore.imageSrc)
const jsonResult = computed(()=>dataStore.jsonResult)
const formattedJson = computed(() =>
  jsonResult.value
    ? JSON.stringify(jsonResult.value, null, 2)
    : '')
function reset(){
  dataStore.setJsonResult(null)
  dataStore.setImageSrc(null)
  ionRouter.navigate('/result', 'forward', 'replace');
}
</script>

<template>
  <div style="display: grid; gap: 12px; max-width: 420px">
    <img
      v-if="imageSrc"
      :src="imageSrc"
      alt="snapshot"
      style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px"
      >
    <h3>JSON Result</h3>
    <pre>{{ formattedJson }}</pre>

    <ion-button @click="reset" style=".cool-btn" ></ion-button>

  </div>


</template>

<style scoped>
.cool-btn {
  --background: linear-gradient(135deg, #000 0%, #fbf6f6 60%);

  --color: black;
  --border-radius: 12px;
  --box-shadow: 0 4px 15px rgba(59, 130, 246, 0.35);
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
.cool-btn::part(native) {
  backdrop-filter: blur(8px);
}
/* Ionicon style override */
ion-icon {
  font-size: 1.2em;
  margin-right: 6px;
}
</style>
