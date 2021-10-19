<script>
    import '../app.css';
    import Header from '$lib/Header.svelte'
</script>

<div class="container">
    <Header />
    <main>
        <slot></slot>
    </main>
</div>

<style>
    .container {
        margin: var(--gap);
        display: grid;
        grid-template-columns: 10fr 2fr;
    }

    main {
        grid-row-start: 2;
    }
</style>