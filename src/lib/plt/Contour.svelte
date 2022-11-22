<script>
  // the code of the countour graph is partially based on https://observablehq.com/@d3/contours
  import * as d3 from "d3";
  import { getContext } from "svelte";

  // for example (x, y) => x ** 2 + y ** 2;
  export let f;
  // e.g. d3.range(-2, 2, 0.1).map((i) => i);
  export let thresholds;

  //linear or log
  export let scale = "linear";

  let xScale = getContext("xScale");
  let yScale = getContext("yScale");
  let domain = getContext("domain");
  let range = getContext("range");
  let x = xScale;
  let y = yScale;

  let color;
  if (scale === "linear") {
    color = d3.scaleSequential(d3.extent(thresholds), d3.interpolateMagma);
  } else if (scale === "log") {
    color = d3.scaleSequentialLog(d3.extent(thresholds), d3.interpolateMagma);
  }

  const x0 = xScale(domain[0]);
  const x1 = xScale(domain[1]);
  const y1 = yScale(range[0]);
  const y0 = yScale(range[1]);
  const n = Math.ceil(x1 - x0);
  const m = Math.ceil(y1 - y0);
  const grid = new Array(n * m);
  for (let j = 0; j < m; ++j) {
    for (let i = 0; i < n; ++i) {
      grid[j * n + i] = f(x.invert(i + x0), y.invert(j + y0));
    }
  }

  let contours = d3.contours().size([n, m]).thresholds(thresholds)(grid);
  let geo = d3.geoPath();
</script>

<g stroke="black" stroke-opacity="0.3" transform="translate({x0}, {y0})">
  {#each contours as contour}
    <path d={geo(contour)} fill={color(contour.value)} />
  {/each}
</g>
