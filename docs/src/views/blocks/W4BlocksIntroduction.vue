<template>
    <main>
        <div class="container">
            <h1>Blocks</h1>
            <h2>How is W4AI organized?</h2>

            <p>World4AI is organized into building <strong>blocks</strong>. Each block builds knowledge that is fundamental to modern artificial intelligence. But you don't need to be proficient in any of the sections to start the next one, therefore you do not need to learn the blocks in a stricly linear manner. You can learn the material that you require as you go along. The sections will generally link to relevant posts from previous sections if the need arises.</p>

            <p class="info">At the moment only a few sections have been started and none is considered finalized. More chapters will be published in the next weeks and months. But the overall process will most likely take many years.</p>

            <img src="../assets/img/blocks/intro/blocks.svg" alt="Blocks">
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
        grid-template-columns: 1fr 1fr 6fr 1fr 1fr;
        grid-template-rows: auto;
        gap: var(--gap);
        justify-content: center;
        align-items: center;
    }

    h1 {
        color: var(--text-color);
        text-transform: uppercase;
        font-weight: 700;
        grid-column-start: 3;
        align-self: center;
        justify-self: left;
        font-size: 70px;
        border-top: 1px solid var(--text-color);
        border-bottom: 1px solid var(--text-color);
    }

    h2 {
        color: var(--text-color);
        grid-column-start: 3;
        text-transform: uppercase;
        font-size: 30px;
        font-weight: 400;
        letter-spacing: 5px;
    }

    p {
        grid-column-start: 3;
        line-height: 2;
        font-size: 18px;
    }

    .info {
        box-shadow: -5px 5px 0px 0px var(--main-color-1), 5px -5px 0px 0px var(--main-color-1);
        padding: 10px;
    }

    img {
        grid-column-start: 4;
        grid-column-end: 6;
        grid-row-start: 3;
    }

</style>