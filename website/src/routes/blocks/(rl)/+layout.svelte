<script>
  import Sidebar from "$lib/Sidebar.svelte";
  import { rl } from "$lib/sidebar_data/data.js";
</script>

<Sidebar root={rl} />
<main>
  <article>
    <p class="danger">The reinforcement learning block is outdated and needs a lot of refactoring.</p>
    <slot />
  </article>
</main>
