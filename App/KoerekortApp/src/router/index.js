import { createRouter, createWebHistory } from '@ionic/vue-router'

import CameraView from '../views/CameraView.vue'
import ResultView from '../views/ResultView.vue'

const routes = [
  { path: '/', component: CameraView },
  { path: '/result', component:ResultView  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router
