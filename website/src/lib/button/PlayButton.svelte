<script>
  // type is either "play" or "pause"
  let type = "play";

  // function that is executed periodically
  export let f;
  // time intervall, function will run once a sec
  export let delta = 1000;

  let intervalId = null;
  function handler() {
    if (!intervalId) {
      intervalId = setInterval(f, delta);
      type = "pause";
    } else {
      clearInterval(intervalId);
      intervalId = null;
      type = "play";
    }
  }
</script>

<button on:click={handler}>
  <!-- the svgs below are feather icons  https://feathericons.com/ -->
  {#if type === "play"}
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      class="feather feather-play-circle"
      ><circle cx="12" cy="12" r="10" /><polygon
        points="10 8 16 12 10 16 10 8"
      /></svg
    >
  {:else if type === "pause"}
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      class="feather feather-pause-circle"
      ><circle cx="12" cy="12" r="10" /><line
        x1="10"
        y1="15"
        x2="10"
        y2="9"
      /><line x1="14" y1="15" x2="14" y2="9" /></svg
    >
  {/if}
</button>

<style>
  button {
    outline: none;
    border: none;
    font-size: 10px;
    cursor: pointer;
    display: flex;
    background-color: transparent;
  }

  button:hover {
    transform: scale(1.02);
  }

  svg {
    fill: var(--main-color-3);
    width: 50px;
    height: 50px;
    stroke-width: 1px;
  }
</style>
