<script>
  import { scaleLinear } from "d3-scale";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";

  let width = 500;
  let height = 200;

  let bias = 0;
  let data = [];

  const xTicks = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50];
  const yTicks = [0, 1];
  const padding = { top: 20, right: 15, bottom: 20, left: 25 };

  function generateData(bias) {
    data = [];
    for (let i = -50; i <= 50; i++) {
      let x = i;
      let y;
      if (x < bias) {
        y = 0;
        data.push({ x, y });
      } else if (x > bias) {
        y = 1;
        data.push({ x, y });
      } else if (x === bias) {
        y = 0;
        data.push({ x, y });
        y = 1;
        data.push({ x, y });
      }
    }
  }

  $: generateData(bias);
  $: minX = data[0].x;
  $: maxX = data[data.length - 1].x;

  $: xScale = scaleLinear()
    .domain([minX, maxX])
    .range([padding.left, width - padding.right]);

  $: yScale = scaleLinear()
    .domain([Math.min.apply(null, yTicks), Math.max.apply(null, yTicks)])
    .range([height - padding.bottom, padding.top]);

  $: path = `M${data.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join("L")}`;
</script>

<svg viewBox="0 0 {width} {height}">
  <!-- y axis -->
  <g class="axis y-axis" transform="translate(0, {padding.top})">
    {#each yTicks as tick}
      <g
        class="tick tick-{tick}"
        transform="translate(0, {yScale(tick) - padding.bottom})"
      >
        <line
          stroke-dasharray="4, 8"
          x2="100%"
          stroke="var(--text-color)"
          opacity="0.2"
        />

        <text y="-4" fill="var(--text-color)">{tick} </text>
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
        <line
          stroke-dasharray="4, 8"
          y1="-{height}"
          y2="-{padding.bottom}"
          x1="0"
          x2="0"
          stroke="var(--text-color)"
          opacity="0.2"
        />
        <text fill="var(--text-color)" y="0" text-anchor="middle">{tick}</text>
      </g>
    {/each}
  </g>
  <path fill="none" stroke="var(--main-color-1)" class="path-line" d={path} />
</svg>

<div class="flex-container">
  <div><Latex>\theta</Latex></div>
  <Slider bind:value={bias} min={-50} max={50} />
</div>

<style>
  .flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .flex-container div {
    width: 30px;
  }
</style>
