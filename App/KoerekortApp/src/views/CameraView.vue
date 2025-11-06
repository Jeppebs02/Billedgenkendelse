<script setup>
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent } from '@ionic/vue'
import CameraComponent from '../components/CameraComponent.vue'
</script>

<template>
  <ion-page>
    <ion-header>
      <ion-toolbar color="dark">
        <div class="toolbar-grid">
          <div></div>
          <ion-title>Driver's License or Nah?</ion-title>
          <div></div>
        </div>
      </ion-toolbar>
    </ion-header>

    <ion-content class="ion-padding">
      <CameraComponent />
    </ion-content>
  </ion-page>
</template>
