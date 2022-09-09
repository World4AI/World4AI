<script>
  import Sidebar from "$lib/Sidebar.svelte";
  import { rl } from "$lib/sidebar_data/data.js";
</script>

<Sidebar root={rl} />
<main>
  <article>
    <slot />
  </article>
</main>
