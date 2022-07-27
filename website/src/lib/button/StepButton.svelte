<script>
  export let disabled = false;
</script>

<button class:disabled {disabled} on:click>
  <!-- the svgs below are feather icons  https://feathericons.com/ -->
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    class="feather feather-skip-forward"
    class:disabled
    ><polygon points="5 4 15 12 5 20 5 4" /><line
      x1="19"
      y1="5"
      x2="19"
      y2="19"
    /></svg
  >
</button>

<style>
  button {
    background-color: var(--background-color);
    outline: none;
    border: none;
    font-size: 10px;
    cursor: pointer;
    display: flex;
  }

  button:hover {
    transform: scale(1.02);
  }

  svg {
    fill: var(--main-color-3);
    width: 50px;
    height: 50px;
    stroke-width: 1px;
    border: 2px solid black;
  }

  .disabled {
    fill: var(--main-color-1);
    stroke: black;
    cursor: wait;
  }
</style>
