<script>
  import { scaleLinear } from "d3-scale";
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

<input name="bias" type="range" bind:value={bias} min="-50" max="50" />

<style>
  input[type="range"] {
    height: 36px;
    -webkit-appearance: none;
    margin: 10px 0;
    width: 100%;
    background-color: var(--background-color);
  }
  input[type="range"]:focus {
    outline: none;
  }
  input[type="range"]::-webkit-slider-runnable-track {
    width: 100%;
    height: 12px;
    cursor: pointer;
    box-shadow: 1px 1px 1px #000000;
    background: var(--text-color);
    border: 1px solid #000000;
  }
  input[type="range"]::-webkit-slider-thumb {
    border: 1px solid #000000;
    height: 28px;
    width: 18px;
    background: var(--main-color-1);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -9px;
  }
  input[type="range"]:focus::-webkit-slider-runnable-track {
    background: var(--text-color);
  }
  input[type="range"]::-moz-range-track {
    width: 100%;
    height: 12px;
    cursor: pointer;
    box-shadow: 1px 1px 1px #000000;
    background: var(--text-color);
    border: 1px solid #000000;
  }
  input[type="range"]::-moz-range-thumb {
    border: 1px solid #000000;
    height: 28px;
    width: 18px;
    background: var(--main-color-1);
    cursor: pointer;
  }
  input[type="range"]::-ms-track {
    width: 100%;
    height: 12px;
    cursor: pointer;
    background: transparent;
    border-color: transparent;
    color: transparent;
  }
  input[type="range"]::-ms-fill-lower {
    background: var(--text-color);
    border: 1px solid #000000;
    box-shadow: 1px 1px 1px #000000;
  }
  input[type="range"]::-ms-fill-upper {
    background: var(--text-color);
    border: 1px solid #000000;
    box-shadow: 1px 1px 1px #000000;
  }
  input[type="range"]::-ms-thumb {
    margin-top: 1px;
    border: 1px solid #000000;
    height: 28px;
    width: 18px;
    background: var(--main-color-1);
    cursor: pointer;
  }
  input[type="range"]:focus::-ms-fill-lower {
    background: var(--text-color);
  }
  input[type="range"]:focus::-ms-fill-upper {
    background: var(--text-color);
  }
</style>
