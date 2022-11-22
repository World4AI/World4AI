<script>
  import { getContext } from "svelte";
  import Latex from "$lib/Latex.svelte";

  export let text = "";
  export let fontSize = 10;

  // choose between "text" and "latex"
  export let type = "text";

  let width = getContext("width");
  let height = getContext("height");

  export let x = width / 2;
  export let y = height - fontSize / 2;
</script>

{#if type === "text"}
  <text class="label" font-size={fontSize} {x} {y}>{text}</text>
{:else if type === "latex"}
  <foreignObject x={0} y={y - fontSize} {width} height="100%">
    <div style="font-size: {fontSize}px">
      <Latex>{text}</Latex>
    </div>
  </foreignObject>
{/if}

<style>
  text.label {
    fill: var(--text-color);
    text-anchor: middle;
    dominant-baseline: middle;
  }

  div {
    display: flex;
    justify-content: center;
  }
</style>
