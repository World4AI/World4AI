<script>
  import { scaleLinear } from "d3-scale";
  let gamma = 0.95;
  let maxX = 100;
  let points = [];

  function createData() {
    points = [];
    for (let x = 0; x <= maxX; x++) {
      let point = { x, y: gamma ** x };
      points.push(point);
    }
  }

  const xTicks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
  const yTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const padding = { top: 20, right: 15, bottom: 20, left: 25 };

  let width = 500;
  let height = 200;

  $: gamma && createData();
  $: minX = points[0].x;
  $: maxX = points[points.length - 1].x;

  $: xScale = scaleLinear()
    .domain([minX, maxX])
    .range([padding.left, width - padding.right]);

  $: yScale = scaleLinear()
    .domain([Math.min.apply(null, yTicks), Math.max.apply(null, yTicks)])
    .range([height - padding.bottom, padding.top]);

  $: path = `M${points.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join("L")}`;
</script>

<div class="svg-container">
  <svg viewBox="0 0 {width} {height}">
    <!-- y axis -->
    <g class="axis y-axis" transform="translate(0, {padding.top})">
      {#each yTicks as tick}
        <g
          class="tick tick-{tick}"
          transform="translate(0, {yScale(tick) - padding.bottom})"
        >
          <line x2="100%" />
          <text y="-4">{tick}</text>
        </g>
      {/each}
    </g>

    <!-- x axis -->
    <g class="axis x-axis">
      {#each xTicks as tick}
        <g
          class="tick tick-{tick}"
          transform="translate({xScale(tick)},{height})"
        >
          <line y1="-{height}" y2="-{padding.bottom}" x1="0" x2="0" />
          <text y="-2">{tick}</text>
        </g>
      {/each}
    </g>

    <!-- data -->
    <path class="path-line" d={path} />
  </svg>
</div>
<form>
  <div>
    <label for="gamma">Gamma:</label>
    <select bind:value={gamma} id="gamma">
      <option value={0.95}>0.95</option>
      <option value={0.99}>0.99</option>
      <option value={0.999}>0.999</option>
    </select>
  </div>
</form>

<style>
  .svg-container {
    max-width: 800px;
  }
  svg {
    width: 100%;
  }

  .tick {
    font-size: 15px;
    font-weight: 200;
  }

  .tick line {
    stroke: var(--text-color);
    stroke-width: 0.1px;
  }

  .tick text {
    fill: var(--text-color);
    text-anchor: start;
  }

  .tick.tick-0 line {
    stroke-dasharray: 0;
  }

  .x-axis .tick text {
    text-anchor: middle;
  }

  .path-line {
    fill: none;
    stroke: var(--main-color-2);
    stroke-linejoin: round;
    stroke-linecap: round;
    stroke-width: 2;
  }

  .path-area {
    fill: rgba(0, 100, 100, 0.2);
  }

  label {
    text-transform: uppercase;
    font-size: 20px;
    margin-right: 15px;
  }

  select {
    padding: 2px;
    border: 1px solid var(--main-color-1);
    outline: none;
    text-align: center;
    cursor: pointer;
    font-size: 20px;
    width: 10%;
    background-color: var(--background-color);
    color: var(--text-color);
  }
</style>
