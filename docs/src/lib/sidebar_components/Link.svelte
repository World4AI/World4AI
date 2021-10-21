<script>
    export let name;
    export let link;
</script>

<a href={link}>{name}</a>
    


<style>
    a {
        text-decoration: none;
        font-size: 15px;
    }

    a:hover {
        color: var(--main-color-1);
    }
</style>