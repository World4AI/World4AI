<script>
  import Sidebar from "$lib/Sidebar.svelte";
  import Alert from "$lib/Alert.svelte";
  import { rl } from "$lib/sidebar_data/data.js";
</script>

<Sidebar root={rl} />
<main>
  <article>
    <Alert type="danger">The reinforcement learning block is outdated and needs a lot of refactoring. </Alert>
    <slot />
  </article>
</main>
