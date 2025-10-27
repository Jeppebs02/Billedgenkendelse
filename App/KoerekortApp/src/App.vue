<template>
  <IonApp>
    <!-- Global header -->
    <IonHeader class="app-header">
      <IonToolbar>
        <div class="header-inner">
          <div class="brand">
            <span class="dot"></span>
            <span class="text">Kørekortsverifivering <span class="grad">Solita</span></span>
          </div>
        </div>
      </IonToolbar>

      <!-- Animated gradient border -->
      <div class="animated-border"></div>
    </IonHeader>

    <!-- Page content -->
    <IonContent class="app-content">
      <IonRouterOutlet />
    </IonContent>

    <!-- Optional footer -->
    <IonFooter class="app-footer">
      <p>© 2025 Kørekortsverifivering <span class="grad">Solita</span></p>
    </IonFooter>
  </IonApp>
</template>

<script setup lang="ts">
import { IonApp, IonHeader, IonToolbar, IonContent, IonRouterOutlet, IonFooter } from '@ionic/vue'
import { onMounted } from 'vue'

onMounted(() => {
  document.title = 'Kørekortsverifivering Solita'
})
</script>

<style scoped>
/* === Solita farvepalette === */
:root {
  --accent-1: #ff0a78;
  --accent-2: #7a3cff;
  --accent-3: #00d0ff;
  --bg: #fafafa;
  --text: #111;
  --line: #e5e5e5;
}

/* === App layout === */
.app-content {
  --background: var(--bg);
  min-height: 100vh;
  color: var(--text);
  font-family: "Inter", system-ui, sans-serif;
  padding-top: 70px;
}

/* === Header === */
.app-header {
  position: fixed;
  top: 0;
  width: 100%;
  z-index: 100;
  background: #fff;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}

.header-inner {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 64px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
  font-size: 16px;
  color: var(--text);
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
  box-shadow: 0 0 0 6px rgba(122, 60, 255, 0.12);
}

.grad {
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

/* === Animated gradient border === */
.animated-border {
  height: 3px;
  background: linear-gradient(270deg, var(--accent-1), var(--accent-2), var(--accent-3), var(--accent-1));
  background-size: 400% 400%;
  animation: gradientMove 6s ease infinite;
  border-radius: 2px;
}

/* Keyframes for the moving gradient */
@keyframes gradientMove {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

/* === Footer === */
.app-footer {
  --background: #fff;
  text-align: center;
  border-top: 1px solid var(--line);
  padding: 12px 0 18px;
  font-size: 13px;
  color: #666;
}
</style>
