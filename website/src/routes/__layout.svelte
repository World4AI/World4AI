<script>
  import "../app.css";
  import Header from "$lib/Header.svelte";
</script>

<div class="container">
  <Header />
  <main>
    <slot />
  </main>
</div>
