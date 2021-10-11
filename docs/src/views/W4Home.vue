<template>
    <main>
        <div class="container">
            <div class="number">1</div>
            <hr>
            <h1>World4AI</h1>
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

.number {
    font-size: 70px;
}

hr {
    height: 1px;
    width: 50%;
    border: 2px solid black;
}

h1 {
    text-transform: uppercase;
    font-weight: 400;
    grid-column-start: 3;
    grid-row-start: 1;
    align-self: center;
    justify-self: left;
    font-size: 70px;
}

img {
    margin-top: 15vh;
    grid-column-start: 3;
    justify-self: right;
}

.container {
    background-image:
      repeating-linear-gradient(rgba(255, 255, 255, 0.1) 0 1px, transparent 1px 100%),
      repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.1) 0 1px, transparent 1px 100%);
    background-size: 30px 30px;
}


</style>