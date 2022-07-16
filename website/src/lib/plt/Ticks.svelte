<script>
  import { getContext } from "svelte";

  export let xTicks = [];
  export let yTicks = [];
  export let showGrid = true;

  let xScale = getContext("xScale");
  let yScale = getContext("yScale");
  let domain = getContext("domain");
  let range = getContext("range");
</script>

<!-- draw axis -->
<g class="axis y-axis">
  {#each yTicks as tick}
    <g class="tick tick-{tick}" transform="translate(0, {yScale(tick)})">
      {#if showGrid}
        <line x1={xScale(domain[0])} x2={xScale(domain[1])} />
      {/if}
      <text x={xScale(domain[0]) - 12} y="+4">{tick}</text>
    </g>
  {/each}
</g>

<g class="axis x-axis">
  {#each xTicks as tick}
    <g class="tick" transform="translate({xScale(tick)},0)">
      {#if showGrid}
        <line y1={yScale(range[0])} y2={yScale(range[1])} />
      {/if}
      <text y={yScale(range[0]) + 12}>{tick}</text>
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
    font-size: 12px;
    fill: var(--text-color);
  }
</style>
