<script>
  import "../../../app.css";
  import Container from "$lib/Container.svelte";
  import Header from "$lib/Header.svelte";
  import Sidebar from "$lib/Sidebar.svelte";
  import { programming } from "$lib/sidebar_data/data.js";

  let isVisible = false;
  function toggleAside() {
    isVisible = !isVisible;
  }
</script>

<div class="container">
  <Header />
  <div class="grid-container">
    <main>
      <Container>
        <slot />
      </Container>
    </main>
    <div on:click={toggleAside} class="aside-toggle" />
    <aside class:visible={isVisible}>
      <Sidebar root={programming} />
    </aside>
  </div>
</div>

<style>
  .container {
    margin: var(--gap);
  }

  .grid-container {
    position: relative;
    display: grid;
    grid-template-columns: 2fr 10fr;
  }

  main {
    grid-column-start: 2;
  }

  aside {
    grid-column-start: 1;
    grid-row-start: 1;
  }

  @media (max-width: 768px) {
    aside {
      position: absolute;
      left: -1000px;
      top: -30px;
    }

    aside.visible {
      transform: translateX(1000px);
    }

    .aside-toggle {
      position: absolute;
      top: -80px;
      left: 40%;
      height: 50px;
      width: 50px;
      border-radius: 50vh;
      background-color: var(--text-color);
      cursor: pointer;
    }

    main {
      grid-column-start: 1;
      grid-column-end: 13;
    }
  }
</style>
