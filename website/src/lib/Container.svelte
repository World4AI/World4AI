<script>
  import { page } from "$app/stores";
  let container;

  function scrollTop() {
    if (container) {
      container.scrollTop = 0;
    }
  }
  // scroll to the top position when the page changes
  $: path = $page.path;
  $: path && scrollTop();
</script>

<div bind:this={container} class="container">
  <slot />
</div>

<style>
  .container {
    margin-top: 50px;
    display: grid;
    grid-template-columns: 1fr 1fr 6fr 1fr 1fr;
    grid-template-rows: auto;
    gap: var(--gap);
    justify-content: center;
    align-items: flex-start;
    color: var(--text-color);
  }

  .container :global(*) {
    grid-column-start: 3;
  }

  .container :global(.heading-large) {
    font-size: 70px;
  }

  .container :global(h1) {
    text-transform: uppercase;
    font-weight: 400;
    justify-self: left;
    font-size: 50px;
  }

  .container :global(h2) {
    text-transform: uppercase;
    font-size: 40px;
    font-weight: 400;
    letter-spacing: 5px;
  }

  .container :global(h3) {
    text-transform: uppercase;
    font-size: 30px;
    font-weight: 400;
    letter-spacing: 5px;
  }

  .container :global(h4) {
    text-transform: uppercase;
    font-size: 25px;
    font-weight: 400;
    letter-spacing: 5px;
  }

  .container :global(p) {
    line-height: 2;
    font-size: 20px;
  }

  .container :global(.flex-center) {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .container :global(.flex-space) {
    display: flex;
    justify-content: space-around;
    align-items: center;
  }

  .container :global(.info) {
    padding-left: 10px;
    border-left: 3px solid var(--text-color);
  }

  .container :global(.separator) {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 10px;
  }

  /* style scrollbar */
  .container {
    max-height: 80vh;
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--aside-color);
  }
  .container::-webkit-scrollbar {
    width: 10px;
  }
  .container::-webkit-scrollbar-thumb {
    background-color: #fff;
  }
  .container::-webkit-scrollbar-track {
    background-color: #000;
  }

  @media (max-width: 768px) {
    .container {
      max-height: 85vh;
      margin-top: 10px;
      grid-template-columns: 6fr;
      grid-template-rows: auto;
      justify-content: center;
      align-items: flex-start;
      color: var(--text-color);
    }

    .container :global(*) {
      grid-column-start: 1;
    }

    .container :global(.heading-large) {
      font-size: 45px;
    }

    .container :global(h1) {
      text-transform: uppercase;
      font-weight: 400;
      justify-self: left;
      font-size: 35px;
    }

    .container :global(h2) {
      text-transform: uppercase;
      font-size: 25px;
      font-weight: 400;
      letter-spacing: 3px;
    }

    .container :global(h3) {
      text-transform: uppercase;
      font-size: 22px;
      font-weight: 400;
      letter-spacing: 3px;
    }

    .container :global(h4) {
      text-transform: uppercase;
      font-size: 20px;
      font-weight: 400;
      letter-spacing: 5px;
    }

    .container :global(p) {
      line-height: 2;
      font-size: 15px;
    }

    .container :global(svg) {
      width: 300px;
      height: 300px;
    }
  }
</style>
