<script>
  import Latex from "$lib/Latex.svelte";
  export let x;
  export let y;
  export let r;
  export let color = "none";
  let className = "";
  export { className as class };

  export let text = "";
  export let type = "text";
  export let fontSize = 7;
</script>

<circle cx={x} cy={y} {r} stroke="black" class={className} fill={color} />

{#if text !== ""}
  {#if type === "text"}
    <text font-size={fontSize} {x} {y}>{text}</text>
  {:else if type === "latex"}
    <foreignObject x={x - r} y={y - r} width={r * 2} height={r * 2}>
      <div
        style="font-size: {fontSize}px; height: {r * 2}px"
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
