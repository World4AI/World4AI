<script>
  export let type = "play";
  export let disabled = false;
</script>

<button class:disabled {disabled} on:click>
  <!-- the svgs below are feather icons  https://feathericons.com/ -->
  {#if type === "play" && !disabled}
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
  {:else if type === "pause" && !disabled}
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
  {:else}
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
      class="feather feather-loader"
      ><line x1="12" y1="2" x2="12" y2="6" /><line
        x1="12"
        y1="18"
        x2="12"
        y2="22"
      /><line x1="4.93" y1="4.93" x2="7.76" y2="7.76" /><line
        x1="16.24"
        y1="16.24"
        x2="19.07"
        y2="19.07"
      /><line x1="2" y1="12" x2="6" y2="12" /><line
        x1="18"
        y1="12"
        x2="22"
        y2="12"
      /><line x1="4.93" y1="19.07" x2="7.76" y2="16.24" /><line
        x1="16.24"
        y1="7.76"
        x2="19.07"
        y2="4.93"
      /></svg
    >
  {/if}
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
    width: 60px;
    height: 60px;
    stroke-width: 1px;
  }

  .disabled {
    color: var(--main-color-1);
    cursor: wait;
  }
</style>
