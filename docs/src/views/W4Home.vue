<template>
    <main>
        <img class="mascot" src="../assets/img/common/mascot.svg" alt="Mascot">
        <div class="text learn">Learn</div>
        <div class="text ai">AI</div>
        <div class="text block-1">Block</div>
        <div class="text by">By</div>
        <div class="text block-2">Block</div>
        <img class="boxes" src="../assets/img/landing/boxes.svg" alt="Boxes">
    </main>
</template>

<script>
export default {

}
</script>

<style scoped>
main {
    min-height: calc(70vh - var(--gap));
    display: grid;
    justify-content: center;
    align-items: center;
    grid-template-columns: repeat(10, 1fr);
    grid-template-rows: repeat(10, 1fr);
    gap: var(--gap);

    /* draw a grid including gaps */
    background-image: 
        linear-gradient(var(--grid-color) 1px, transparent 0), 
        linear-gradient(90deg, var(--grid-color) 1px, transparent 0),
        linear-gradient(var(--grid-color) 1px, transparent 0),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 0);
    background-size: 
        calc((83.33vw - var(--gap)) / 10) calc((70vh) / 10), 
        calc((83.33vw - var(--gap)) / 10) calc((70vh) / 10), 
        calc((83.33vw - var(--gap)) / 10) calc((70vh) / 10),
        calc((83.33vw - var(--gap)) / 10) calc((70vh) / 10);
    background-position: 0px 0px, 0px 0px, 0px calc(-1 * var(--gap)), calc(-1 * var(--gap)) 0px;
}


.mascot {
    justify-self: center;
    align-self: flex-end;
    grid-column-start: 2;
    grid-row-start: 2;
    grid-column-end: 2;
    grid-row-end: 7;
    width: 80%;
}

.text {
    text-transform: uppercase;
    font-weight: 700;
    font-size: 1.8em;
}

.text .learn {
    grid-column-start: 1;
    grid-row-start: 1;
}

.text.ai {
    grid-column-start: 1;
    grid-row-start: 2;
}

.text.block-1 {
    grid-column-start: 3;
    grid-row-start: 5;
}

.text.by {
    grid-column-start: 3;
    grid-row-start: 6;
}

.text.block-2 {
    grid-column-start: 3;
    grid-row-start: 7;
}

.boxes {
    grid-column-start: 5;
    grid-column-end: 10;
    grid-row-start: 5;
    grid-row-end: 10;
    align-self: flex-end;
    justify-self: flex-end;
}
</style>