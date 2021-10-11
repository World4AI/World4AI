<template>
    <main>
        <div class="container">
            <div class="number">1</div>
            <hr>
            <h1>Learn AI Block by Block</h1>
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
    font-size: 50px;
}

hr {
    height: 1px;
    width: 50%;
    border: 2px solid black;
}

h1 {
    text-transform: uppercase;
    grid-column-start: 3;
    grid-row-start: 1;
    align-self: center;
    justify-self: right;
    font-size: 50px;
}

</style>