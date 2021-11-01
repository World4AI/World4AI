<div class="container">
    <slot></slot>
</div>

<style>
    .container {
        margin-top: 50px;
        display: grid;
        grid-template-columns: 1fr 1fr 6fr 1fr 1fr;
        grid-template-rows: auto;
        gap: var(--gap);
        justify-content: center;
        align-items: flex-start;
        color: var(--text-color);
    }

    .container :global(*)  {
        grid-column-start: 3;
    }

    .container :global(.heading-large){
        font-size: 70px;
    }

    .container :global(h1) {
        text-transform: uppercase;
        font-weight: 400;
        justify-self: left;
        font-size: 50px;
    }

    .container :global(h2) {
        grid-column-start: 3;
        text-transform: uppercase;
        font-size: 40px;
        font-weight: 400;
        letter-spacing: 5px;
    }

    .container :global(h3) {
        text-transform: uppercase;
        font-size: 30px;
        font-weight: 400;
        letter-spacing: 5px;
    }

    .container :global(h4) {
        text-transform: uppercase;
        font-size: 25px;
        font-weight: 400;
        letter-spacing: 5px;
    }

    .container :global(p) {
        line-height: 2;
        font-size: 20px;
    }

    .container :global(.flex-center) {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .container :global(.info) {
        position: relative;
    }

    .container :global(.info::before) {
        content: url(/icons/info-outline.svg);
        left: -70px;
        top: 0px;
        position: absolute;
        height: 40px;
        width: 40px;
    }

    /* style scrollbar */
    .container {
        max-height: 80vh;
        overflow: auto;
        scrollbar-width: thin;
        scrollbar-color: var(--aside-color);
    }
    .container::-webkit-scrollbar {
        width: 10px;
    }
    .container::-webkit-scrollbar-thumb {
        background-color: #FFF;
    }
    .container::-webkit-scrollbar-track {
        background-color: #000;
    }


</style>