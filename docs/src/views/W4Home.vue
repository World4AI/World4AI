<template>
    <main>
        <div class="container background">
            <h1>World4AI</h1>
            <h2>Advance In AI Block By Block</h2>
            <img src="../assets/img/landing/boxes.svg" alt="">
        </div>
    </main>
</template>

<script>
export default {

}
</script>

<style scoped>
.container {
    display: grid;
    grid-template-columns: 0.5fr 0.5fr 3fr 0.5fr 0.5fr;
    grid-template-rows: auto;
    gap: var(--gap);
    justify-content: center;
    align-items: center;
}
.background {
    background-image:
      repeating-linear-gradient(rgba(255, 255, 255, 0.1) 0 1px, transparent 1px 100%),
      repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.1) 0 1px, transparent 1px 100%);
    background-size: 30px 30px;
}

h1 {
    text-transform: uppercase;
    font-weight: 400;
    grid-column-start: 3;
    align-self: center;
    justify-self: left;
    font-size: 70px;
}

h2 {
    grid-column-start: 3;
}

img {
    margin-top: 15vh;
    grid-column-start: 3;
    justify-self: right;
}
</style>