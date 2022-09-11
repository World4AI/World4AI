<script>
  import { getContext } from "svelte";

  export let data;
  export let color = "var(--main-color-1)";
  export let size = 5;

  let xScale = getContext("xScale");
  let yScale = getContext("yScale");
</script>

{#each data as point}
  <rect fill={color} x={xScale(point.x) - size/2} y={yScale(point.y)-size/2} height={size} width={size} />
{/each}

<style>
  rect {
    stroke: var(--text-color);
    stroke-width: 1px;
  }
</style>
