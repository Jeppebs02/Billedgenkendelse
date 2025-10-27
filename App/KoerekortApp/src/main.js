import './assets/main.css'

import { createApp } from 'vue'
import { IonicVue } from '@ionic/vue';
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(IonicVue)
app.use(router)

router.isReady().then(() => {
  app.mount('#app');
});
