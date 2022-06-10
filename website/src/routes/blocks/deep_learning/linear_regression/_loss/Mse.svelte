<script>
  import Scatterplot from "$lib/Scatterplot.svelte";
  import Plot from "$lib/Plot.svelte";
  import Latex from "$lib/Latex.svelte";

  export let data = [
    [
      { x: 5, y: 20 },
      { x: 10, y: 40 },
      { x: 35, y: 15 },
      { x: 45, y: 59 },
    ],
  ];

  export let w = 0;
  export let b = 0;

  let mse = 0;
  let regressionLine = [];
  let lines = [];
  let rectangles = [];

  function linesAndBoxes(w, b) {
    regressionLine = [];
    lines = [];
    rectangles = [];
    mse = 0;

    //regression line
    let x1 = 0;
    let y1 = b + w * x1;
    let x2 = 60;
    let y2 = b + w * x2;
    let line = [
      { x: x1, y: y1 },
      { x: x2, y: y2 },
    ];
    regressionLine.push(line);
    lines.push(line);
    rectangles.push(line);

    data[0].forEach((point) => {
      //lines
      let x1 = point.x;
      let x2 = point.x;
      let y1 = point.y;
      let y2 = b + w * x2;

      let line = [
        { x: x1, y: y1 },
        { x: x2, y: y2 },
      ];
      lines.push(line);

      //sum squred error
      mse += (y1 - y2) ** 2;

      //rectangles
      let x = point.x;
      let y = y1 > y2 ? y1 : y2;
      let size = Math.abs(y1 - y2);
      let rect = [
        { x, y },
        { x: x + size, y },
        { x: x + size, y: y - size },
        { x, y: y - size },
        { x, y },
      ];
      rectangles.push(rect);
    });
    mse = mse / data[0].length;
  }

  $: linesAndBoxes(w, b);
</script>

<span class="plot-container">
  <div class="calculations">
    <p><Latex>{String.raw`w`}</Latex>: {w.toFixed(2)}</p>
    <p><Latex>{String.raw`b`}</Latex>: {b.toFixed(2)}</p>
    <p><Latex>{String.raw`MSE`}</Latex>: {mse.toFixed(2)}</p>
  </div>
  <Plot
    pointsData={data}
    pathsData={rectangles}
    config={{
      width: 500,
      height: 500,
      maxWidth: 600,
      minX: 0,
      maxX: 60,
      minY: 0,
      maxY: 60,
      xLabel: "Feature",
      yLabel: "Label",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 5,
      colors: ["var(--main-color-1)", "var(--main-color-2)"],
      numTicks: 7,
    }}
  />
</span>

<style>
  .plot-container {
    position: relative;
  }

  .calculations {
    padding: 10px 15px;
    position: absolute;
    left: 10px;
    width: 150px;
    background-color: var(--main-color-4);
    z-index: 1;
  }

  .calculations > p {
    margin: 0;
    font-size: 15px;
  }
</style>
