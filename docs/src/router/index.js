import {createRouter, createWebHistory} from 'vue-router'
import W4Home from '../views/W4Home.vue'

const routes = [
    {
        path: '/',
        name: 'Home',
        component: W4Home
    },
    {
        path: '/blocks/math/introduction',
        name: 'Math Introduction',
        component: () => import ('../views/W4MathIntroduction.vue')
    },
    {
        path: '/blocks/programming/introduction',
        name: 'Programming Introduction',
        component: () => import ('../views/W4ProgrammingIntroduction.vue')
    },
    {
        path: '/blocks/deep_learning/introduction',
        name: 'Deep Learning Introduction',
        component: () => import ('../views/W4DeepLearningIntroduction.vue')
    },
    {
        path: '/blocks/reinforcement_learning/introduction',
        name: 'Programming Introduction',
        component: () => import ('../views/W4ReinforcementLearningIntroduction.vue')
    },
    {
        path: '/support',
        name: 'Support',
        component: () => import ('../views/W4Support.vue')
    },
    {
        path: '/about',
        name: 'About',
        component: () => import ('../views/W4About.vue')
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router