<script>
  import Latex from "$lib/Latex.svelte";
  export let width;
  export let height;
  export let x;
  export let y;
  export let text = "";
  export let type = "text";

  export let color = "none";
  export let fontSize = 7;
</script>

<rect
  x={x - width / 2}
  y={y - height / 2}
  {width}
  {height}
  stroke="black"
  fill={color}
/>

{#if text}
  {#if type === "text"}
    <text font-size={fontSize} {x} {y}>{text}</text>
  {:else if type === "latex"}
    <foreignObject x={x - width / 2} y={y - height / 2} {width} {height}>
      <div
        style="font-size: {fontSize}px; height: {height}px"
        class="flex justify-center items-center"
      >
        <Latex>{text}</Latex>
      </div>
    </foreignObject>
  {/if}
{/if}

<style>
  text {
    dominant-baseline: middle;
    text-anchor: middle;
  }
</style>
