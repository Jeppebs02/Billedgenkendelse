<script setup>
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent } from '@ionic/vue'
import ResultComponent from '../components/ResultComponent.vue'

</script>

<template>
  <ion-page>
    <ion-header>
      <ion-toolbar color="dark">
        <ion-title>Result</ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content class="ion-padding">
      <ResultComponent />
    </ion-content>
  </ion-page>
</template>
