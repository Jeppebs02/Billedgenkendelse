import { createRouter, createWebHistory } from '@ionic/vue-router'
import HomeView from '../views/HomeView.vue'
import CameraView from '../views/CameraView.vue'


const routes = [
  { path: '/', component: HomeView },
  { path: '/camera', component: CameraView },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router
