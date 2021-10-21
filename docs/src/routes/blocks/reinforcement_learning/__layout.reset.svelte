<script>
    import '../../../app.css';
    import Container from '$lib/Container.svelte'
    import Header from '$lib/Header.svelte';
    import Sidebar from '$lib/Sidebar.svelte';
</script>

<div class="container">
    <div class="left-container">
        <Header></Header>
        <main>
            <Container>
                <slot></slot>
            </Container>
        </main> 
    </div>
    <div class="rigth-container">
        <Sidebar></Sidebar>
    </div>
</div>
    
<style>
    .container {
        display: grid;
        grid-template-columns: 10fr 2fr;
    }

    .left-container {
        margin: var(--gap);
    }

    main {
        grid-row-start: 3;
    }

</style>