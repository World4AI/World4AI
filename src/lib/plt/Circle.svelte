<script>
  import { getContext } from "svelte";

  export let data;
  export let color = "var(--main-color-1)";
  export let radius = 5;

  let xScale = getContext("xScale");
  let yScale = getContext("yScale");
</script>

{#each data as point}
  <circle fill={color} cx={xScale(point.x)} cy={yScale(point.y)} r={radius} />
{/each}

<style>
  circle {
    stroke: var(--text-color);
    stroke-width: 1px;
  }
</style>
