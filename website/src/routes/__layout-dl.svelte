<script>
  import "../app.css";
  import Header from "$lib/Header.svelte";
  import Sidebar from "$lib/Sidebar.svelte";
  import { dl } from "$lib/sidebar_data/data.js";

  let showToc = false;

  function swithToc() {
    showToc = !showToc;
  }
</script>

<div class="overlay" class:active={showToc} />
<div class="container">
  <Header />
  <div class="flex-container" class:active={showToc}>
    <div class="flex-left">
      <div class="sidebar-container" class:active={showToc}>
        <div class="arrow" on:click={swithToc} />
        <Sidebar root={dl} />
      </div>
    </div>
    <div class="flex-right">
      <main>
        <article>
          <slot />
        </article>
      </main>
    </div>
  </div>
</div>

<style>
  .flex-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
  }

  .flex-left {
    flex-basis: 300px;
    flex-grow: 0;
    flex-shrink: 0;
    position: relative;
  }

  .flex-right {
    flex-grow: 1;
  }

  .sidebar-container {
    z-index: 1;
    position: fixed;
    top: 250px;
    left: 0px;
    transition: all 1s;
  }

  .overlay {
    display: none;
  }

  .arrow {
    display: none;
  }

  @media (max-width: 1000px) {
    .flex-container.active {
      overflow: hidden;
      max-height: 100vh;
    }

    .flex-left {
      flex-basis: 0;
    }

    .sidebar-container {
      top: 120px;
      transform: translateX(-300px);
    }

    .sidebar-container.active {
      transform: translateX(0px);
    }

    .overlay {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0px;
      right: 0;
      background-color: var(--text-color);
    }

    .overlay.active {
      display: block;
    }

    .arrow {
      position: fixed;
      left: 290px;
      top: 0px;
      width: 30px;
      height: 30px;

      border: solid var(--main-color-1);
      border-width: 0 3px 3px 0;
      display: inline-block;
      padding: 3px;
      transition: all 1sec;

      transform: rotate(-45deg);
      -webkit-transform: rotate(-45deg);
    }

    .arrow:hover {
      cursor: pointer;
    }
  }
</style>
