<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import * as d3 from "d3";

  // the code of the countour graph is partially based on https://observablehq.com/@d3/contours
  let width = 500;
  let height = 500;

  // for example (x, y) => x ** 2 + y ** 2;
  export let value;
  // for example (x, y) => {return { x: 2 * x, y: 2 * y };};
  export let gradient;
  export let epochs = 25;

  let minX = -210;
  let minY = -210;
  let maxX = 210;
  let maxY = 210;

  let thresholds = d3.range(1, 19).map((i) => Math.pow(2, i));
  let color = d3.scaleSequentialLog(d3.extent(thresholds), d3.interpolateMagma);
  let x = d3.scaleLinear([minX, maxX], [0, width]);
  let y = d3.scaleLinear([minY, maxY], [height, 0]);

  const x0 = 0;
  const x1 = width;
  const y0 = 0;
  const y1 = height;
  const n = Math.ceil(x1 - x0);
  const m = Math.ceil(y1 - y0);
  const grid = new Array(n * m);
  for (let j = 0; j < m; ++j) {
    for (let i = 0; i < n; ++i) {
      grid[j * n + i] = value(x.invert(i + x0), y.invert(j + y0));
    }
  }

  let contours = d3.contours().size([n, m]).thresholds(thresholds)(grid);
  let geo = d3.geoPath();

  let xAxis = d3.range(-180, 190, 30);
  let yAxis = d3.range(-180, 190, 30);

  //gradient descent
  let coordinates = [];
  let startingX = 190;
  let startingY = 190;

  function calculateGradients() {
    coordinates = [];
    for (let i = 0; i < epochs; i++) {
      let x1;
      let x2;
      let y1;
      let y2;
      if (i === 0) {
        x1 = startingX;
        y1 = startingY;
      } else {
        x1 = coordinates[i - 1].x2;
        y1 = coordinates[i - 1].y2;
      }
      let grads = gradient(x1, y1);
      x2 = x1 - 0.1 * grads.x;
      y2 = y1 - 0.1 * grads.y;
      let coordinate = { x1, y1, x2, y2 };
      coordinates.push(coordinate);
    }
  }
  calculateGradients();

  function handleClick(e) {
    let rect = e.target.getBoundingClientRect();
    let width = rect.width;
    let height = rect.height;

    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    let scaleX = d3.scaleLinear().domain([0, width]).range([minX, maxX]);
    let scaleY = d3.scaleLinear().domain([0, height]).range([maxY, minY]);

    x = scaleX(x);
    y = scaleY(y);

    startingX = x;
    startingY = y;
    calculateGradients();
  }
</script>

<SvgContainer maxWidth="600px">
  <svg on:click={handleClick} viewBox="0 0 {width} {height}">
    <g stroke="black" stroke-opacity="0.3">
      {#each contours as contour}
        <path d={geo(contour)} fill={color(contour.value)} />
      {/each}
    </g>
    {#each coordinates as coordinate}
      <line
        x1={x(coordinate.x1)}
        y1={y(coordinate.y1)}
        x2={x(coordinate.x2)}
        y2={y(coordinate.y2)}
        stroke="black"
        stroke-opacity="0.5"
        stroke-dasharray="5 10"
      />
    {/each}
    <circle
      cx={x(coordinates[0].x1)}
      cy={y(coordinates[0].y1)}
      r="7"
      fill="var(--main-color-4)"
      stroke="var(--text-color)"
    />
    {#each coordinates as coordinate}
      <circle
        cx={x(coordinate.x2)}
        cy={y(coordinate.y2)}
        r="2"
        fill="var(--main-color-4)"
        stroke="black"
      />
    {/each}

    <!-- x axis -->
    <g class="x-axis">
      {#each xAxis as axis}
        <line
          x1={x(axis)}
          x2={x(axis)}
          y1={height}
          y2={height - 5}
          stroke="black"
        />
        <text x={x(axis)} y={height - 10}>{axis}</text>
      {/each}
    </g>
    <!-- y axis -->
    <g class="y-axis">
      {#each yAxis as axis}
        <line x1={0} x2={0 + 5} y1={y(axis)} y2={y(axis)} stroke="black" />
        <text x={20} y={y(axis)}>{axis}</text>
      {/each}
    </g>
  </svg>
</SvgContainer>

<style>
  * {
    pointer-events: none;
  }

  svg {
    cursor: pointer;
    pointer-events: all;
  }
  text {
    dominant-baseline: middle;
    text-anchor: middle;
    font-size: 10px;
    vertical-align: middle;
    display: inline-block;
  }
</style>
