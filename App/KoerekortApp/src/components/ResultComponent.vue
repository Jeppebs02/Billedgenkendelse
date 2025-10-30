<script setup>
import { computed } from 'vue'
import { IonButton, useIonRouter } from '@ionic/vue'
import { useDataStore } from '@/stores/dataStore.js'

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
  <div class="responsive-grid">
    <div class="card">
      <img
        v-if="imageSrc"
        :src="imageSrc"
        alt="snapshot"
        class="picture-style"
        >
      <h3>JSON Result</h3>
      <pre>{{ formattedJson }}</pre>

      <ion-button class="cool-btn" @click="reset">Reset</ion-button>
    </div>
  </div>
</template>
