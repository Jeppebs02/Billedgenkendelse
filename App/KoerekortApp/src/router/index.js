import { createRouter, createWebHistory } from '@ionic/vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
  },
  {
    path: '/about',
    name: 'about',
    component: () => import('../views/AboutView.vue'),
  },
  {
    path: '/camera',
    name: 'camera',
    component: () => import('../views/CameraView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router
