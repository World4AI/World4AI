<script>
  // the code is based on the official Svelte examples
  import { scaleLinear } from "d3-scale";
  import SvgContainer from "./SvgContainer.svelte";

  export let pathsData = [];
  export let pointsData = [];
  export let heatmapData = [];

  //deal with cases when you only have one category of points and paths
  $: if (pointsData[0] && !Array.isArray(pointsData[0])) {
    pointsData = [[...pointsData]];
  }

  $: if (pathsData[0] && !Array.isArray(pathsData[0])) {
    pathsData = [[...pathsData]];
  }

  export let config = {};
  const standardConfig = {
    width: 500,
    height: 250,
    maxWidth: 1000,
    minX: 0,
    maxX: 1,
    minY: 0,
    maxY: 1,
    xLabel: "Input",
    yLabel: "Output",
    padding: { top: 20, right: 40, bottom: 40, left: 60 },
    radius: 5,
    colors: ["var(--main-color-1)", "var(--main-color-2)", "var(--text-color)"],
    heatmapColors: ["var(--main-color-3)", "var(--main-color-4)"],
    xTicks: [],
    yTicks: [],
    numTicks: 5,
  };
  config = { ...standardConfig, ...config };

  const heatmapSize = config.width / Math.sqrt(heatmapData.length);

  function createTicks(min, max) {
    let ticks = [];

    for (let i = 0; i < config.numTicks; i++) {
      let result = min + ((max - min) / (config.numTicks - 1)) * i;

      // show only fractions if you don't deal with integers
      if (Math.floor(result) < result) {
        result = result.toFixed(2);
      }
      ticks.push(result);
    }
    return ticks;
  }

  let xScale = scaleLinear()
    .domain([config.minX, config.maxX])
    .range([config.padding.left, config.width - config.padding.right]);

  let yScale = scaleLinear()
    .domain([config.minY, config.maxY])
    .range([config.height - config.padding.bottom, config.padding.top]);

  // in case ticks are not provided we make a best guess based on number of desired ticks
  if (config.xTicks.length === 0) {
    config.xTicks = createTicks(config.minX, config.maxX);
  }

  if (config.yTicks.length === 0) {
    config.yTicks = createTicks(config.minY, config.maxY);
  }

  $: paths = [];
  $: pathsData.forEach((data) => {
    paths = [];
    let path = `M${data.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join("L")}`;
    paths.push(path);
  });
</script>

<SvgContainer maxWidth="{config.maxWidth}px">
  <svg viewBox="0 0 {config.width} {config.height}">
    <!-- heatmap -->
    {#each heatmapData as coordinate}
      <rect
        x={xScale(coordinate.x) - heatmapSize / 2}
        y={yScale(coordinate.y) - heatmapSize / 2}
        width={heatmapSize}
        height={heatmapSize}
        fill={config.heatmapColors[coordinate.class]}
        stroke-width="0.05px"
        stroke={config.heatmapColors[coordinate.class]}
        opacity="0.4"
      />
    {/each}
    <!-- draw axis -->
    <g class="axis y-axis">
      {#each config.yTicks as tick}
        <g class="tick tick-{tick}" transform="translate(0, {yScale(tick)})">
          <line
            class:zero={tick === 0}
            x1={config.padding.left}
            x2={xScale(config.maxX)}
          />
          <text x={config.padding.left - 8} y="+4">{tick}</text>
        </g>
      {/each}
    </g>

    <g class="axis x-axis">
      {#each config.xTicks as tick}
        <g class="tick" transform="translate({xScale(tick)},0)">
          <line
            class:zero={tick === 0}
            y1={yScale(config.minY)}
            y2={yScale(config.maxY)}
          />
          <text y={config.height - config.padding.bottom + 16}>{tick}</text>
        </g>
      {/each}
    </g>

    <!-- Add labels -->
    <text
      class="label"
      x={xScale((config.maxX - config.minX) / 2 + config.minX)}
      y={config.height - config.padding.bottom + 35}>{config.xLabel}</text
    >

    <text class="label" transform="rotate(-90)" x={-config.height / 2} y={15}
      >{config.yLabel}</text
    >

    <!-- paths -->
    {#each paths as path}
      <path d={path} />
    {/each}

    <!-- points -->
    {#each pointsData as pointCategory, categoryIdx}
      {#each pointCategory as point}
        <circle
          fill={config.colors[categoryIdx]}
          cx={xScale(point.x)}
          cy={yScale(point.y)}
          r={config.radius}
        />
      {/each}
    {/each}
  </svg>
</SvgContainer>

<style>
  circle {
    stroke: rgba(0, 0, 0, 0.5);
  }

  path {
    fill: none;
    stroke: var(--text-color);
    stroke-linejoin: round;
    stroke-linecap: round;
    stroke-width: 1;
  }
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
    stroke-width: 2px;
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
