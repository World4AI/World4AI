<script>
  import "../app.css";
  import Header from "$lib/Header.svelte";
  import Container from "$lib/Container.svelte";
</script>

<div class="container">
  <Header />
  <main>
    <Container>
      <article>
        <slot />
      </article>
    </Container>
  </main>
</div>
