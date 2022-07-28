<script>
  import Folder from "$lib/sidebar_components/Folder.svelte";
  export let root;

  let tocActive = true;

  function toggleToc() {
    tocActive = !tocActive;
  }
</script>

<aside class:move-left={!tocActive}>
  <Folder name="ROOT" links={root} />
</aside>
<div class="toc-control" class:move-left={!tocActive} on:click={toggleToc}>
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="30"
    height="30"
    viewBox="0 0 24 24"
    fill="var(--background-color)"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    class="feather feather-book-open toc"
    ><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" /><path
      d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"
    /></svg
  >
</div>

<style>
  aside {
    position: fixed;
    margin-top: 100px;
    left: 0px;
    top: 0px;
    max-width: 350px;
    background-color: var(--main-color-3);
    padding: 20px;
    max-height: calc(100vh - 100px);
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--text-color);
    z-index: 2;
    transition: all 1s;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
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

  .toc-control {
    position: fixed;
    margin-top: 100px;
    top: 10px;
    left: 300px;
    z-index: 3;
    transition: all 1s;
  }

  .toc:hover {
    cursor: pointer;
  }

  .move-left {
    transform: translateX(-350px);
  }

  .toc-control.move-left {
    transform: translateX(-280px);
  }
  @media (max-width: 1000px) {
    aside {
      margin-top: 70px;
      max-height: calc(100vh - 70px);
    }
    .toc-control {
      margin-top: 70px;
    }
  }
  @media (max-width: 300px) {
    .toc-control {
      left: 210px;
    }
    .toc-control.move-left {
      transform: translateX(-210px);
    }
  }
</style>
