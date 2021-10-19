<script>
    import { page } from '$app/stores';
    $: active = $page.path
</script>

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

<style>
    header {
        display: flex;
        justify-content: space-between;
        align-items: center;
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