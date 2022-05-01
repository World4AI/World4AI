<script>
  import { page } from "$app/stores";
  export let name;
  export let link;

  $: path = $page.url.pathname;
</script>

<a class:selected={path === link} href={link}>{name}</a>

<style>
  a {
    display: inline-block;
    text-decoration: none;
    font-size: 15px;
    color: var(--text-color);
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
    left: -20px;
    background-color: var(--text-color);
  }
</style>
