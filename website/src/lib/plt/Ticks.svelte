<script>
  import { getContext } from "svelte";

  export let xTicks = [];
  export let yTicks = [];
  export let showGrid = true;
  export let fontSize = 12;

  let xScale = getContext("xScale");
  let yScale = getContext("yScale");
  let domain = getContext("domain");
  let range = getContext("range");
  let height = getContext("height");
</script>

<!-- draw axis -->
<g class="axis y-axis">
  {#each yTicks as tick}
    <g class="tick tick-{tick}" transform="translate(0, {yScale(tick)})">
      {#if showGrid}
        <line x1={xScale(domain[0])} x2={xScale(domain[1])} />
      {/if}
      <text font-size={fontSize} x={fontSize}>{tick}</text>
    </g>
  {/each}
</g>

<g class="axis x-axis">
  {#each xTicks as tick}
    <g class="tick" transform="translate({xScale(tick)},0)">
      {#if showGrid}
        <line y1={yScale(range[0])} y2={yScale(range[1])} />
      {/if}
      <text font-size={fontSize} y={height - fontSize}>{tick}</text>
    </g>
  {/each}
</g>

<style>
  .tick line {
    stroke: var(--text-color);
    stroke-dasharray: 2 2;
    opacity: 0.4;
  }

  line {
    stroke: var(--text-color);
    opacity: 0.4;
  }

  line.zero {
    stroke-dasharray: none;
    opacity: 1;
  }

  .axis {
    dominant-baseline: middle;
    text-anchor: middle;
  }

  text {
    fill: var(--text-color);
  }
</style>
