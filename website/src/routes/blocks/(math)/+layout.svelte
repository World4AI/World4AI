<script>
  import Sidebar from "$lib/Sidebar.svelte";
  import { math } from "$lib/sidebar_data/data.js";
</script>

<Sidebar root={math} />
<main>
  <article>
    <slot />
  </article>
</main>
