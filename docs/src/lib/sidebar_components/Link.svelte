<script>
    import { page } from '$app/stores';
    export let name;
    export let link;

    $: path = $page.path;
</script>

<a class:selected={path===link} href={link}>{name}</a>
    
<style>
    a {
        text-decoration: none;
        font-size: 15px;
        color: #FFF;
    }

    .selected {
        color: var(--main-color-1);
    }

    a:hover {
        color: var(--main-color-1);
    }
</style>