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
  ionRouter.navigate('/', 'forward', 'replace');
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

    <ion-button class="cool-btn" @click="reset">Try with a new photo</ion-button>

  </div>
</template>
