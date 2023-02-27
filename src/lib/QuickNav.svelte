<script>
  import { ArrowRightCircle, ArrowLeftCircle } from "lucide-svelte";
  import { page } from "$app/stores";

  export let links;

  let idx;
  $: path = $page.url.pathname;
  $: {
    links.forEach((link, i) => {
      if (path === link) {
        idx = i;
      }
    });
  }
</script>

<div
  class={`container mx-auto flex ${
    idx !== 0 ? "justify-between" : "justify-end"
  }`}
>
  {#if idx !== 0}
    <a aria-label="previous-page" href={links[idx - 1]}>
      <ArrowLeftCircle />
    </a>
  {/if}
  {#if idx !== links.length - 1}
    <a aria-label="next-page" href={links[idx + 1]}>
      <ArrowRightCircle />
    </a>
  {/if}
</div>
