<script>
  export let data;

  import { page } from "$app/stores";
  $: path = $page.url.pathname;
</script>

<aside
  class="bg-gray-100 select-none border-r sticky top-16 left-0 max-h-[92vh] overflow-y-auto z-40"
>
  <nav class="p-4">
    {#each data as folder}
      <h6 class="font-bold text-sm">{folder.name}</h6>
      <ul class="text-sm mb-4">
        {#each folder.links as section}
          <li>
            <a
              class:border-red-400={path === section.link}
              class="border-l inline-block py-2 px-4 w-full hover:bg-gray-200"
              href={section.link}>{section.name}</a
            >
          </li>
        {/each}
      </ul>
    {/each}
  </nav>
</aside>

<style>
  /* style scrollbar */
  aside {
    scrollbar-width: thin;
    scrollbar-color: var(--text-color);
  }

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
