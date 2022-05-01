<script>
  import Folder from "$lib/sidebar_components/Folder.svelte";
  export let root;
</script>

<aside class="mx-3">
  <Folder name="ROOT" links={root} />
</aside>

<style>
  aside {
    width: 250px;
    background-color: var(--main-color-4);
    padding: 20px;
    max-height: 70vh;
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--text-color);
  }
  /* style scrollbar */
  aside::-webkit-scrollbar {
    width: 5px;
  }
  aside::-webkit-scrollbar-thumb {
    background-color: var(--text-color);
  }
  aside::-webkit-scrollbar-track {
    background-color: var(--background-color);
  }
</style>
