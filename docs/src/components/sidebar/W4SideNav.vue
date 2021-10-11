<template>
    <nav>
        <W4SideNavGroup @toggle-visibility="toggleVisibility" :elements='elements' />
    </nav>
</template>

<script>
import W4SideNavGroup from './W4SideNavGroup.vue'
import navigation from '../../assets/data/navigation.json'

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
            elements: navigation
        }
    },

}
</script>

<style scoped>

</style>