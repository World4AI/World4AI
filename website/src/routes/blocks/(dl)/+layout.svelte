<script>
  import Sidebar from "$lib/Sidebar.svelte";
  import { dl } from "$lib/sidebar_data/data.js";
</script>

<Sidebar root={dl} />
<main>
  <article>
    <slot />
  </article>
</main>
