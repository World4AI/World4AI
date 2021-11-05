<script>
    import '../../../app.css';
    import Container from '$lib/Container.svelte'
    import Header from '$lib/Header.svelte';
    import Sidebar from '$lib/Sidebar.svelte';
    import {programming} from '$lib/sidebar_data/data.js'
</script>

<div class="container">
    <Header></Header>
    <div class="grid-container">
        <main>
            <Container>
                <slot></slot>
            </Container>
        </main>
        <aside>
            <Sidebar root={programming}></Sidebar>
        </aside>
    </div>
</div>
    
<style>
    .container {
        margin: var(--gap);
    }
    
    .grid-container {
        display: grid;
        grid-template-columns: 2fr 10fr;
    }

    main {
        grid-column-start: 2;
    }

    aside{
        grid-column-start: 1;
        grid-row-start: 1;
    }
</style>