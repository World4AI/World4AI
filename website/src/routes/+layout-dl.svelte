<script>
  import "../app.css";
  import Header from "$lib/Header.svelte";
  import Sidebar from "$lib/Sidebar.svelte";
  import { dl } from "$lib/sidebar_data/data.js";
</script>

<Header />
<Sidebar root={dl} />
<main>
  <article>
    <slot />
  </article>
</main>
