<script>
  export let width;
  export let height;
  export let x;
  export let y;
  export let text;

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
  <text font-size={fontSize} {x} {y}>{text}</text>
{/if}

<style>
  text {
    dominant-baseline: middle;
    text-anchor: middle;
  }
</style>
