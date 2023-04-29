<script>
  import { page } from "$app/stores";
  import Container from "$lib/Container.svelte";
</script>

<Container>
  <h1>Page Not Found: 404</h1>
  <div class="separator" />
  <p>
    {#if $page.status === 404}
      The page that you are looking for does not exist.
    {:else}
      An unexpected error occurred.
    {/if}
  </p>
</Container>
