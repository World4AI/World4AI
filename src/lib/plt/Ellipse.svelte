<script>
  import { getContext } from "svelte";

  export let data;
  export let color = "var(--main-color-1)";
  export let radiusX = 5;
  export let radiusY = 5;

  let xScale = getContext("xScale");
  let yScale = getContext("yScale");
</script>

{#each data as point}
  <ellipse
    fill={color}
    cx={xScale(point.x)}
    cy={yScale(point.y)}
    rx={radiusX}
    ry={radiusY}
  />
{/each}

<style>
  ellipse {
    stroke: var(--text-color);
    stroke-width: 1px;
  }
</style>
