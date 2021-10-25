<script>
    import Link from "$lib/sidebar_components/Link.svelte";

    export let name;
    export let links;
</script>

{#if name !== 'ROOT'}
    <span>{name}</span>
{/if}

<ul>
    {#each links as link}
            <li>
            {#if link.links}
                <svelte:self {...link} />
            {:else}
                <div class="box">
                    <Link {...link}/>
                </div>
            {/if} 
        </li>
    {/each}
</ul>

<style>
    ul {
        list-style: none;
    }

    li {
        font-size: 20px;
        margin: 5px 0px;
        letter-spacing: 2px;
    }

    .box {
        margin-left: 20px;
    }

    span {
        color: var(--text-color);
        display: inline-block;
        text-transform: uppercase;
        border-bottom: 1px dotted rgba(0, 0, 0, .3);
        margin: 10px 0;
    }
</style>

