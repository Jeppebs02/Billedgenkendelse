<script setup>
import { computed } from 'vue'
import { IonButton, useIonRouter } from '@ionic/vue'
import { useDataStore } from '@/stores/dataStore.js'

const dataStore = useDataStore()
const ionRouter = useIonRouter()
const imageSrc = computed(()=>dataStore.imageSrc)
const jsonResult = computed(()=>dataStore.jsonResult)


//only failed checks
const failedChecks = computed(() =>(jsonResult.value?.checks || []).filter(check => !check.passed))
const header = computed(() => (jsonResult.value.decision === "APPROVED"))

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
      <h3 v-if="header">Driver license ready!</h3>
      <h3 v-else>Nahhhh</h3>
      <div>
        <ul class="checks-list">
          <li v-for="(check, index) in failedChecks" :key="index">
            ❌ {{ check.message }}
          </li>
        </ul>
      </div>

      <ion-button class="cool-btn" @click="reset">Try with a new photo</ion-button>
    </div>
  </div>
</template>
