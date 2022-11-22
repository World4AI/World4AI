<script>
  import { getContext } from "svelte";
  import Latex from "$lib/Latex.svelte";

  export let text = "Output";
  export let fontSize = 10;

  // choose between "text" and "latex"
  export let type = "text";

  let width = getContext("width");
  let height = getContext("height");

  export let x = fontSize / 2;
  export let y = height / 2;
</script>

{#if type === "text"}
  <text
    class="label"
    font-size={fontSize}
    transform="rotate(-90 {x} {y})"
    {x}
    {y}>{text}</text
  >
{:else if type === "latex"}
  <foreignObject {x} y={0} {width} {height}>
    <div
      style="font-size: {fontSize}px; transform-origin: {x + fontSize}px {y}px"
    >
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
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transform: rotate(-90deg);
  }
</style>
