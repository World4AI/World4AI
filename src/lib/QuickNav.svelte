<script>
  import { page } from "$app/stores";
  export let links;

  $: path = $page.url.pathname;
</script>

<div class="container mx-auto flex justify-between">
  {#each links as link, idx}
    {#if path === link}
      {#if idx !== 0}
        <a href={links[idx - 1]}>
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
            class="feather feather-arrow-left-circle"
            ><circle cx="12" cy="12" r="10" /><polyline
              points="12 8 8 12 12 16"
            /><line x1="16" y1="12" x2="8" y2="12" /></svg
          >
        </a>
      {/if}
      {#if idx !== links.length - 1}
        <a href={links[idx + 1]}>
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
            class="feather feather-arrow-right-circle"
            ><circle cx="12" cy="12" r="10" /><polyline
              points="12 16 16 12 12 8"
            /><line x1="8" y1="12" x2="16" y2="12" /></svg
          >
        </a>
      {/if}
    {/if}
  {/each}
</div>
