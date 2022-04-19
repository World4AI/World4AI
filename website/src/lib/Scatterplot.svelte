<script>
  // the code is mostly from the official Svelte examples
  import { scaleLinear } from "d3-scale";
  import SvgContainer from "./SvgContainer.svelte";

  export let data = [];

  export let colors = ["var(--main-color-1)", "var(--main-color-2)"];

  export let width = 500;
  export let height = 250;
  export let maxWidth;
  export let xLabel = "Feature 1";
  export let yLabel = "Feature 2";

  export let lines = [];
  export let rectangles = [];

  export let minX = 0;
  export let maxX = 1;
  export let minY = 0;
  export let maxY = 1;
  export let radius = 5;

  export let numTicks = 2;

  function createTicks(min, max) {
    let ticks = [];

    for (let i = 0; i < numTicks; i++) {
      ticks.push(min + Math.floor((max - min) / (numTicks - 1)) * i);
    }
    return ticks;
  }

  const padding = { top: 20, right: 40, bottom: 40, left: 60 };

  $: xScale = scaleLinear()
    .domain([minX, maxX])
    .range([padding.left, width - padding.right]);

  $: yScale = scaleLinear()
    .domain([minY, maxY])
    .range([height - padding.bottom, padding.top]);

  $: heightScale = scaleLinear()
    .domain([minY, maxY])
    .range([padding.top, height - padding.bottom]);

  $: xTicks = createTicks(minX, maxX);
  $: yTicks = createTicks(minY, maxY);
</script>

<SvgContainer {maxWidth}>
  <svg viewBox="0 0 {width} {height}">
    <g class="axis y-axis">
      {#each yTicks as tick}
        <g class="tick tick-{tick}" transform="translate(0, {yScale(tick)})">
          <line x1={padding.left} x2={xScale(maxX)} />
          <text x={padding.left - 8} y="+4">{tick}</text>
        </g>
      {/each}
    </g>

    <g class="axis x-axis">
      {#each xTicks as tick}
        <g class="tick" transform="translate({xScale(tick)},0)">
          <line y1={yScale(minY)} y2={yScale(maxY)} />
          <text y={height - padding.bottom + 16}>{tick}</text>
        </g>
      {/each}
    </g>

    <!-- add line to the scatterplot -->
    {#each lines as line}
      <line
        x1={xScale(line.x1)}
        y1={yScale(line.y1)}
        x2={xScale(line.x2)}
        y2={yScale(line.y2)}
      />
    {/each}

    <!-- add rectangles -->
    {#each rectangles as rect}
      {console.log(yScale(rect.height))}
      <rect
        x={xScale(rect.x)}
        y={yScale(rect.y)}
        width={xScale(rect.width) - padding.left}
        height={heightScale(rect.height) - padding.top}
      />
    {/each}
    <!-- Add labels -->
    <text
      class="label"
      x={xScale((maxX - minX) / 2 + minX)}
      y={height - padding.bottom + 35}>{xLabel}</text
    >

    <text class="label" transform="rotate(-90)" x={-height / 2} y={15}
      >{yLabel}</text
    >

    <!-- data -->
    {#each data as points, idx}
      {#each points as point}
        <circle
          fill={colors[idx]}
          cx={xScale(point.x)}
          cy={yScale(point.y)}
          r={radius}
        />
      {/each}
    {/each}
  </svg>
</SvgContainer>

<style>
  circle {
    stroke: rgba(0, 0, 0, 0.5);
  }

  rect {
    stroke: var(--text-color);
    fill: none;
    stroke-dasharray: 2 4;
  }

  .tick line {
    stroke: var(--text-color);
    stroke-dasharray: 4 8;
    opacity: 0.4;
  }

  line {
    stroke: var(--text-color);
  }

  text {
    font-size: 12px;
    fill: var(--text-color);
  }

  text.label {
    font-size: 17px;
    text-anchor: middle;
  }

  .x-axis text {
    text-anchor: middle;
  }

  .y-axis text {
    text-anchor: end;
  }
</style>
