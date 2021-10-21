<script>
    import { page } from '$app/stores';
    
    $: path = $page.path
    $: subPath =(() => {
        return '/' + $page.path.split('/')[1]
    })()
</script>

<header>
    <a href="/"><img class="logo" src="/logo/logo.svg" alt="World4AI Logo"></a>
    <nav>
        <ul>
            <li><a class:selected={subPath === '/'} href="/">Home</a></li>
            <li class="dropdown">
                <a class:selected={subPath === '/blocks'} href="/blocks/introduction">Blocks</a>
                <div class="dropdown-content">
                    <ul>
                        <li><a class:selected={path === '/blocks/introduction'} href="/blocks/introduction">Introduction</a></li>
                        <li><a class:selected={path === '/blocks/mathematics'} href="/blocks/mathematics">Mathematics</a></li>
                        <li><a class:selected={path === '/blocks/programming'} href="/blocks/programming">Programming</a></li>
                        <li><a class:selected={path === '/blocks/deep_learning'} href="/blocks/deep_learning">Deep Learning</a></li>
                        <li><a class:selected={path === '/blocks/reinforcement_learning'} href="/blocks/reinforcement_learning">Reinforcement Learning</a></li>
                    </ul>
                </div>
            </li>
            <li><a class:selected={subPath === '/about'} href="/about">About</a></li>
            <li><a class:selected={subPath === '/sponsor'} href="/sponsor">Sponsor</a></li>
        </ul>
    </nav>

</header>

<style>
    header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .dropdown {
        position: relative;
    }

    .dropdown-content {
        position: absolute;
        display: none;
    }

    .dropdown-content ul {
        margin-top: 20px;
        background-color: #FFF;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }

    .dropdown-content li {
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .dropdown:hover .dropdown-content {
        display: block;
    }

    .logo {
        height: 30px;
        width: 30px;
    }

    nav > ul {
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