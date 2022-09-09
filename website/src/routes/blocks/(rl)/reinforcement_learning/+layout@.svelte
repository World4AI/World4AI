<script>
  import "../../../../app.css";
  import Header from "$lib/Header.svelte";
  import Sidebar from "$lib/Sidebar.svelte";
  import { rl } from "$lib/sidebar_data/data.js";
</script>

<Header />
<Sidebar root={rl} />
<main>
  <article>
    <slot />
  </article>
</main>
