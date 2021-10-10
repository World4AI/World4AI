<template>
  <ul class="root">
      <li v-bind:key='element.id' v-for='element in elements'>
        <span v-if="visible">
            <router-link class="hoverable" v-if="!element.children" :to='element.link'>{{element.text}}</router-link>
            <button v-else class="hoverable" @click="$emit('toggle-visibility', element.id)">{{element.text}}</button>
            <W4SideNavGroup @toggle-visibility='toggleVisibility' class="group" :visible='element.visible' v-if="element.children" :elements='element.children' />
        </span>
      </li>
  </ul>
</template>

<script>

export default {
    methods: {
        toggleVisibility(id) {
            this.$emit('toggle-visibility', id)
        }
    },
    props: {
        elements: Array,
        visible: {
            type: Boolean,
            default: true
        }
    }
}
</script>

<style scoped>

    .hoverable:hover {
        color: var(--main-color-1);
    }

    .root {
        position: relative;
    }

    .root::before {
        content: '';
        position: absolute;
        left: -20px;
        top: 0;
        height: 100%;
        width: 0px;
        border-left: 1px dotted rgba(255, 255, 255, 0.2);
    }

    li {
        margin-bottom: 1rem;
        position: relative;
    }

    .group {
        margin-top: 1rem;
        margin-left: 1rem;
    }

    a, li, button {
        letter-spacing: 5px;
        color: #FFF;
        text-transform: uppercase;
        text-decoration: none;
        list-style: none;
        border: none;
        background: none;
        font-size: 1rem;
        text-align: left;
    }

    a, button {
        position: relative;
    }

    a::before, button::before, button::after {
        content: '';
        bottom: 8px;
        left: -20px;
        position: absolute;
        height: 1px;
        width: 20px;
        border-top: 1px dotted rgba(255, 255, 255, 0.2);
    }

    button::after {
        content: '';
        position: absolute;
        left: -24px;
        bottom: 6px;
        height: 0;
        width: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;  
        border-top: 5px solid var(--main-color-2);

    }

    button {
        cursor: pointer;
    }

</style>