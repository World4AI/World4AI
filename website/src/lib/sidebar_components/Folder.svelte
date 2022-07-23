<script>
  import { page } from "$app/stores";

  export let name;
  export let link = "";
  export let links = [];
  export let level = 0;

  $: path = $page.url.pathname;
</script>

{#if name !== "ROOT"}
  <span
    ><a
      class:selected={path === link}
      class:folder={links.length > 0}
      class:intermediate={links.length > 0 && level > 1}
      href={link}>{name}</a
    ></span
  >
{/if}

<ul>
  {#each links as link}
    <li>
      <div class:box={name !== "ROOT"}>
        <svelte:self {...link} level={level + 1} />
      </div>
    </li>
  {/each}
</ul>

<style>
  ul {
    list-style: none;
    margin: 0;
  }

  li {
    font-size: 20px;
    letter-spacing: 2px;
    margin: 5px 0px;
  }

  .box {
    margin-left: 20px;
  }

  a {
    text-decoration: none;
    color: var(--text-color);
    display: inline-block;
    font-size: 15px;
  }

  .selected {
    position: relative;
  }

  .selected::before {
    position: absolute;
    content: "";
    height: 5px;
    width: 5px;
    top: 7px;
    left: -15px;
    background-color: var(--text-color);
  }

  .folder {
    border-bottom: 1px dashed rgba(0, 0, 0, 0.2);
    text-transform: uppercase;
    margin: 7px 0px;
  }

  .folder.intermediate {
    border: none;
    margin: 0px;
    text-transform: capitalize;
  }
</style>
