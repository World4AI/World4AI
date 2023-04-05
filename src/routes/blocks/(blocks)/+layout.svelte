<script>
  import Sidebar from "$lib/Sidebar.svelte";
  import QuickNav from "$lib/QuickNav.svelte";
  import { page } from "$app/stores";

  export let data;
  let showSidebar = false;

  $: path = $page.url.pathname;
  $: block = path.split("/")[2];
  $: nav = data[block];

  let links;
  $: {
    links = [];
    data[block].forEach((chapter) => {
      chapter.links.forEach((link) => {
        links.push(link.link);
      });
    });
  }
</script>

<div class="container mx-auto mb-2">
  <div class="py-2 xl:hidden sticky top-12 left-0 right-0 z-50">
    <button on:click={() => (showSidebar = !showSidebar)}>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="feather feather-menu"
        ><line x1="3" y1="12" x2="21" y2="12" /><line
          x1="3"
          y1="6"
          x2="21"
          y2="6"
        /><line x1="3" y1="18" x2="21" y2="18" /></svg
      >
    </button>
  </div>

  <div class:hidden={!showSidebar} class="xl:hidden">
    <Sidebar data={nav} />
  </div>
  <div class:hidden={showSidebar} class="xl:grid xl:grid-cols-5 gap-2">
    <div class="hidden xl:block">
      <Sidebar data={nav} />
    </div>
    <main class="lg:col-span-4">
      <article>
        <slot />
      </article>
      <QuickNav {links} />
    </main>
  </div>
</div>
