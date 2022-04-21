<script>
  import Scatterplot from "$lib/Scatterplot.svelte";

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
    rectangles = [];
    mse = 0;

    //regression line
    let x1 = 0;
    let y1 = b + w * x1;
    let x2 = 60;
    let y2 = b + w * x2;
    let line = { x1, x2, y1, y2 };
    regressionLine.push(line);
    lines.push(line);

    data[0].forEach((point) => {
      //lines
      let x1 = point.x;
      let x2 = point.x;
      let y1 = point.y;
      let y2 = b + w * x2;

      //sum squred error
      mse += (y1 - y2) ** 2;

      //rectangles
      let x = point.x;
      let y = y1 > y2 ? y1 : y2;
      let width = Math.abs(y1 - y2);
      let rect = { x, y, width, height: width };
      rectangles.push(rect);
    });
    mse = mse / data[0].length;
  }

  $: linesAndBoxes(w, b);
</script>

<Scatterplot
  maxWidth={"600px"}
  width={500}
  height={500}
  {data}
  minX={0}
  maxX={60}
  mixY={0}
  maxY={60}
  numTicks={7}
  lines={regressionLine}
  {rectangles}
  xLabel={"Feature"}
  yLabel={"Label"}
/>
