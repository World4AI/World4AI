
<div class="container">
    <h1>World4AI</h1>
    <h2>Advance In AI Block By Block</h2>
    <img src="landing/boxes.svg" alt="">
</div>

<style>
    .container {
        margin-top: 50px;
        display: grid;
        grid-template-columns: 1fr 1fr 6fr 1fr 1fr;
        grid-template-rows: auto;
        gap: var(--gap);
        justify-content: center;
        align-items: center;
    }

    .container > * {
        grid-column-start: 3;
    }

    h1 {
        color: var(--text-color);
        text-transform: uppercase;
        font-weight: 700;
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
    img {
        margin-top: 25vh;
        grid-column-start: 3;
        justify-self: right;
    }
</style>