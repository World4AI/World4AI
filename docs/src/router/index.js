import {createRouter, createWebHistory} from 'vue-router'
import W4Home from '../views/W4Home.vue'

const routes = [
    {
        path: '/',
        component: W4Home
    },
    {
        path: '/blog',
        component: () => import ('../views/W4Blog.vue')
    },
    {
        path: '/blocks/introduction',
        component: () => import ('../views/W4BlocksIntroduction.vue')
    },
    {
        path: '/blocks/math/introduction',
        component: () => import ('../views/W4MathIntroduction.vue')
    },
    {
        path: '/blocks/programming/introduction',
        component: () => import ('../views/W4ProgrammingIntroduction.vue')
    },
    {
        path: '/blocks/deep_learning/introduction',
        component: () => import ('../views/W4DeepLearningIntroduction.vue')
    },
    {
        path: '/blocks/reinforcement_learning/introduction',
        component: () => import ('../views/W4ReinforcementLearningIntroduction.vue')
    },
    {
        path: '/support',
        component: () => import ('../views/W4Support.vue')
    },
    {
        path: '/about',
        component: () => import ('../views/W4About.vue')
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router