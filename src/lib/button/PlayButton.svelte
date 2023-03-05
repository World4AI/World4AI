<script>
  import { PlayCircle, PauseCircle } from "lucide-svelte";
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
    <PlayCircle
      aria-label="Start the animation"
      class="fill-w4ai-yellow"
      size={40}
      strokeWidth={1}
    />
  {:else if type === "pause"}
    <PauseCircle
      aria-label="Start the animation"
      class="fill-w4ai-yellow"
      size={40}
      strokeWidth={1}
    />
  {/if}
</button>
