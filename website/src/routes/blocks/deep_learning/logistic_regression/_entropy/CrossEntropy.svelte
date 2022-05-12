<script>
  /* inspired by https://svelte.dev/examples/bar-chart */

  import { scaleLinear } from "d3-scale";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Slider from "$lib/Slider.svelte";

  export let maxWidth = "800px";

  export let points1 = [
    { event: "x1", percentage: 0.2 },
    { event: "x2", percentage: 0.2 },
    { event: "x3", percentage: 0.2 },
    { event: "x4", percentage: 0.2 },
    { event: "x5", percentage: 0.2 },
  ];

  export let startingPoints = [
    { event: "x1", percentage: 0.4 },
    { event: "x2", percentage: 0.4 },
    { event: "x3", percentage: 0.05 },
    { event: "x4", percentage: 0.05 },
    { event: "x5", percentage: 0.1 },
  ];

  export let yTicks = [0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

  let points2;
  let crossEntropy;
  let fraction = 0;

  function recalculate(fraction) {
    // recalculate points 2
    points2 = JSON.parse(JSON.stringify(startingPoints));
    for (let i = 0; i < points1.length; i++) {
      points2[i].percentage =
        startingPoints[i].percentage +
        (points1[i].percentage - startingPoints[i].percentage) * fraction;
    }
    // recalculate cross-entropy
    crossEntropy = 0;
    for (let i = 0; i < points1.length; i++) {
      if (points2[i].percentage != 0) {
        crossEntropy +=
          -points1[i].percentage * Math.log2(points2[i].percentage);
      }
    }
  }

  $: recalculate(fraction);

  const padding = { top: 20, right: 0, bottom: 20, left: 35 };

  let width = points1.length * 100;
  let height = 200;

  $: xScale = scaleLinear()
    .domain([0, points1.length])
    .range([padding.left, width - padding.right]);

  $: yScale = scaleLinear()
    .domain([0, Math.max.apply(null, yTicks)])
    .range([height - padding.bottom, padding.top]);

  $: innerWidth = width - (padding.left + padding.right);
  $: barWidth = innerWidth / points1.length / 2.2;
</script>

<SvgContainer {maxWidth}>
  <svg viewBox="0 0 {width} {height}">
    <!-- y axis -->
    <g class="axis y-axis">
      {#each yTicks as tick}
        <g class="tick tick-{tick}" transform="translate(0, {yScale(tick)})">
          <line x2="100%" />
          <text y="-4">{tick} </text>
        </g>
      {/each}
    </g>

    <!-- x axis -->
    <g class="axis x-axis">
      {#each points1 as point, i}
        <g class="tick" transform="translate({xScale(i)},{height})">
          <text x={barWidth / 2 + 20} y="-4">{point.event}</text>
        </g>
      {/each}
    </g>

    <g class="bars">
      {#each points1 as point, i}
        <rect
          class="rect-1"
          x={xScale(i)}
          y={yScale(point.percentage)}
          width={barWidth - 4}
          height={yScale(0) - yScale(point.percentage)}
        />
      {/each}
      {#each points2 as point, i}
        <rect
          class="rect-2"
          x={xScale(i) + barWidth}
          y={yScale(point.percentage)}
          width={barWidth - 4}
          height={yScale(0) - yScale(point.percentage)}
        />
      {/each}
    </g>
  </svg>
</SvgContainer>
<div class="cross-entropy">
  <span>Cross-Entropy: </span>{crossEntropy.toFixed(4)}
</div>
<Slider min="0" max="1" step="0.01" bind:value={fraction} />

<style>
  .tick {
    font-size: 0.725em;
    font-weight: 600;
  }

  .tick line {
    stroke: black;
    stroke-dasharray: 4 4;
    stroke-width: 0.5px;
  }

  .tick text {
    fill: black;
    text-anchor: start;
  }

  .tick.tick-0 line {
    stroke-dasharray: 0;
  }

  .x-axis .tick text {
    text-anchor: middle;
  }

  .bars rect {
    stroke: black;
  }

  .rect-1 {
    fill: var(--main-color-1);
  }

  .rect-2 {
    fill: var(--main-color-3);
  }

  .cross-entropy {
    font-size: 25px;
    width: fit-content;
    background-color: var(--main-color-2);
    padding: 7px 5px;
    font-weight: 600;
  }
</style>
