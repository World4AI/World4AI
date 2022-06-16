<script>
  import "../app.css";
  import Header from "$lib/Header.svelte";
</script>

<Header />
<main>
  <article>
    <slot />
  </article>
</main>
