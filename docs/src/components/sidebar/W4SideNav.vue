<template>
    <nav>
        <W4SideNavGroup @toggle-visibility="toggleVisibility" :elements='elements' />
    </nav>
</template>

<script>
import W4SideNavGroup from './W4SideNavGroup.vue'

export default {
    components: {
        W4SideNavGroup
    },
    methods: {
        toggleVisibility(id) {
            this.elements = this.traverseTree(id, this.elements)
        },
        traverseTree(id, oldArray){
            let newArray = oldArray.map((element) => {
                if (id === element.id) {
                    return {...element, visible: !element.visible}
                }
                if (id !== element.id && !element.children) {
                    return element
                }
                if (id !== element.id && element.children){
                    return {...element, children: this.traverseTree(id, element.children)}
                }
            })
            return newArray
        }
    },
    data() {
        return{
            elements: [
                {
                    id: 1,
                    text: 'Home',
                    link: '/'
                },
                {
                    text: 'Blocks',
                    id: 2,
                    visible: false,
                    children: [
                        {
                            id: 3,
                            text: 'Math',
                            visible: false,
                            children: [
                                {
                                    id: 4,
                                    text: 'Introduction',
                                    link: '/blocks/math/introduction'
                                }
                            ]
                        },
                        {
                            id: 5,
                            text: 'Programming',
                            visible: false,
                            children: [
                                {
                                    id: 6,
                                    text: 'Introduction',
                                    link: '/blocks/programming/introduction'
                                }
                            ]
                        },
                        {
                            id: 7,
                            text: 'Deep Learning',
                            visible: false,
                            children: [
                                {
                                    id: 8,
                                    text: 'Introduction',
                                    link: '/blocks/deep_learning/introduction'
                                }
                            ]
                        },
                        {
                            id: 9,
                            text: 'Reinforcement Learning',
                            visible: false,
                            children: [
                                {
                                    id: 10,
                                    text: 'Introduction',
                                    link: '/blocks/reinforcement_learning/introduction'
                                },
                            ]
                        }
                    ]
                },
                {
                    id: 11,
                    text: 'Support',
                    link: '/support'
                },
                {
                    id: 12, 
                    text: 'About',
                    link: '/about'
                }
            ]
        }
    }
}
</script>

<style scoped>

</style>