<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import { scaleLinear } from "d3-scale";
  export let width = 500;
  export let height = 200;

  let pathLength = 100;
  let numberOfPaths = 100;
  //generate ticks for the x axis
  const xTicks = [];
  const yTicks = [-15, -10, -5, 0, 5, 10, 15];
  const padding = { top: 20, right: 15, bottom: 20, left: 25 };
  for (let i = 0; i < pathLength + 1; i++) {
    if (i % 10 === 0) {
      xTicks.push(i);
    }
  }

  //genrate paths
  let monteCarloPaths = [];
  for (let p = 0; p < numberOfPaths; p++) {
    let path = [];
    let lastPathValue = 0;
    path.push({ x: 0, y: lastPathValue });
    for (let i = 0; i < pathLength; i++) {
      // create process using random variables between - 1 and 1
      lastPathValue += Math.random() * 2 - 1;
      let entry = { x: i + 1, y: lastPathValue };
      path.push(entry);
    }
    monteCarloPaths.push(path);
  }

  let minX = monteCarloPaths[0][0].x;
  let maxX = monteCarloPaths[0][monteCarloPaths[0].length - 1].x;

  let xScale = scaleLinear()
    .domain([minX, maxX])
    .range([padding.left, width - padding.right]);
  let yScale = scaleLinear()
    .domain([Math.min.apply(null, yTicks), Math.max.apply(null, yTicks)])
    .range([height - padding.bottom, padding.top]);

  let paths = [];
  function generatePaths() {
    for (let p = 0; p < numberOfPaths; p++) {
      let path = `M${monteCarloPaths[p]
        .map((p) => `${xScale(p.x)},${yScale(p.y)}`)
        .join("L")}`;
      paths.push(path);
    }
  }

  generatePaths();
</script>

<SvgContainer maxWidth="700px">
  <svg viewBox="0 0 {width} {height}">
    <g class="y" transform="translate(0, {padding.top})">
      {#each yTicks as tick}
        <g transform="translate(0, {yScale(tick) - padding.bottom})">
          <line x2="100%" />
          <text transform="translate(3, -5)">{tick}</text>
        </g>
      {/each}
    </g>
    {#each xTicks as tick}
      <g class="x" transform="translate({xScale(tick)},{height})">
        <text y="0">{tick}</text>
        <line y1="-{height}" y2="-{padding.bottom}" x1="0" x2="0" />
      </g>
    {/each}
    <g fill="none" stroke="var(--text-color)" stroke-opacity="0.2">
      {#each paths as path}
        <path d={path} />
      {/each}
    </g>
  </svg>
</SvgContainer>

<style>
  line {
    stroke: var(--text-color);
    stroke-width: 0.1px;
  }

  text {
    stroke: none;
    fill: var(--text-color);
    font-size: 12px;
  }
  .x text {
    text-anchor: middle;
  }
</style>
