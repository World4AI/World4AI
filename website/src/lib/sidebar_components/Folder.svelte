<script>
    import Link from "$lib/sidebar_components/Link.svelte";

    export let name;
    export let links;
</script>

{#if name !== 'ROOT'}
    <span>{name}</span>
    <hr>
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
        padding: 10px 0;
    }

    hr {
        border: none;
        background: rgba(255, 255, 255, 0.1);
        height: 1px;
    }
</style>

