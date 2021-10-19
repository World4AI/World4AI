<script>
    import '../app.css';
    import { page } from '$app/stores';

    $: active = $page.path
</script>

<div class="container">
    <header>
        <a href="/"><img class="logo" src="/logo/logo.svg" alt="World4AI Logo"></a>
        <nav>
            <ul>
                <li><a class:selected={active === '/'} href="/">Home</a></li>
                <li><a class:selected={active === '/blocks'} href="/blocks">Blocks</a></li>
                <li><a class:selected={active === '/about'} href="/about">About</a></li>
                <li><a class:selected={active === '/sponsor'} href="/sponsor">Sponsor</a></li>
            </ul>
        </nav>
    </header>
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

    header {
        grid-column-start: 1;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    main {
        grid-column-start: 1;
        grid-row-start: 2;
    }

    .logo {
        height: 30px;
        width: 30px;
    }

    nav ul {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    li {
        list-style: none;
        margin: 0 20px;
        letter-spacing: 5px;
    }

    a {
        color: black;
        text-decoration: none;
        text-transform: uppercase;
    }

    a:hover {
        color: var(--main-color-1);
    }

    .selected {
        color: var(--main-color-1);
    }

</style>