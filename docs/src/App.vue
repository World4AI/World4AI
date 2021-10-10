<template>
    <div class="grid-container">
        <div class="left-container">
            <W4Header />
            <div class="main-wrapper">
                <router-view class="main"/>
            </div>
        </div>
        <div class="right-container">
            <W4Aside />
        </div>
    </div>
</template>

<script>
import W4Header from './components/W4Header.vue'
import W4Main from './views/W4Home.vue'
import W4Aside from './components/W4Aside.vue'

export default {
    components: {
        W4Header,
        W4Main,
        W4Aside
    }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --background-color: #E7B859;
    --aside-color: #302F32;
    --main-color-1: #FF683C;
    --main-color-2: #4EB6D7;
    --grid-color: rgba(255, 255, 255, 0.2);
    --gap: 20px;
}

body {
    font-family: 'Montserrat', Helvetica, sans-serif;
}

.grid-container {
    height: 100%;
    display: grid;
    grid-template-columns: 10fr 2fr;
}

.left-container {
    height: 100%;
    background-color: var(--background-color);
}

.right-container {
    height: 100%;
    background-color: var(--aside-color);
}

.main-wrapper {
    margin: 0 var(--gap) var(--gap) var(--gap);
}

</style>