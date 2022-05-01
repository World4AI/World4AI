<script>
  import Question from "$lib/Question.svelte";
  import Scatterplot from "$lib/Scatterplot.svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";

  let data = [
    [
      { x: 5, y: 20 },
      { x: 10, y: 40 },
      { x: 35, y: 15 },
      { x: 45, y: 59 },
    ],
  ];

  let w = 1;
  let b = 0;
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
    let line = { x1, x2, y1, y2 };
    regressionLine.push(line);
    lines.push(line);

    data[0].forEach((point) => {
      //lines
      let x1 = point.x;
      let x2 = point.x;
      let y1 = point.y;
      let y2 = b + w * x2;
      let line = { x1, x2, y1, y2 };
      lines.push(line);

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

<svelte:head>
  <title>World4AI | Deep Learning | Linear Regression Loss</title>
  <meta
    name="description"
    content="Linear regression utilizes the mean squared error as a measure of succuss or failure. The goal is to minimize the MSE."
  />
</svelte:head>

<h1>Loss Function</h1>
<Question>How do we measure success and failure?</Question>
<div class="separator" />

<p>
  While we can intuitively tell how far away our line is from some optimal
  location, we need a quantitative measure that the linear regression algorithm
  can use for optimization purposes. In machine learning we use a measure called
  loss (or loss function). Using this quantity we can tweak the weights and
  biases to find a minimum value of the loss.
</p>
<p>
  In our example we will use only 4 data points. A bigger dataset would
  otherwise clutter the illustrations.
</p>
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
  xLabel={"Feature"}
  yLabel={"Label"}
/>
<p>
  We will start by drawing a 45 degree line from the (0,0) to the (60, 60)
  position. While this looks "OK", we do net have a way to compare that
  particular line with any other lines.
</p>
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
  xLabel={"Feature"}
  yLabel={"Label"}
/>

<p>
  The first step is to calculate the error between the actual label <Latex
    >y_i</Latex
  > and the predicted value <Latex>{String.raw`\hat{y}_i=b+wx_i`}</Latex> for each
  datapoint
  <Latex>i</Latex>. We can define the difference <Latex
    >{String.raw`y_i - \hat{y}_i`}</Latex
  > as the error. Visually we can draw that error as the line that connects the regression
  line with the true label.
</p>
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
  {lines}
  xLabel={"Feature"}
  yLabel={"Label"}
/>
<p>
  If we tried to sum up all the errors in the dataset <Latex
    >{String.raw`\sum_i^n \hat{y}_i - y_i`}</Latex
  > we would realize that the positive and the negative errors are offsetting each
  other. If the errors were symmetrical above and below the line, we would end up
  with a summed error of 0.
</p>
<p>
  The loss that used in linear regression, the mean squared error (MSE), takes
  the error to the power of 2 to get rid of the negative sign. The error is then
  divided by the number of data points to calculate the mean (average) of the
  error. The mean squared error is mathematically expressed as follows.
</p>
<Latex>{String.raw`\Large MSE=\frac{1}{n}\sum_i^n (\hat{y} - y_i)^2`}</Latex>

<p>
  Visually it actually makes sense to imagine squares. Each data point has a
  corresponding square and the larger the area of that square, the larger the
  contribution to the MSE. The length of the side of the square is calculated as
  the absolute distance between <Latex>y</Latex> and <Latex
    >{String.raw`\hat{y}`}</Latex
  >. Try to use the example below and move the weight and the bias. Observe how
  the mean squared error changes based on the parameters.
</p>
<div class="flex-group">
  <div class="scatter">
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
  </div>
  <div class="text">
    <div class="separator" />
    <p><Latex>w</Latex> is: {w}</p>
    <div class="separator" />
    <p><Latex>b</Latex> is: {b}</p>
    <div class="separator" />
    <p><Latex>MSE</Latex> is: {mse}</p>
    <div class="separator" />
  </div>
</div>
<Slider bind:value={w} min={-20} max={20} step={0.1} />
<Slider bind:value={b} min={-50} max={50} />
<p>
  Different combination of the weight <Latex>w</Latex> and the bias <Latex
    >b</Latex
  > produce different losses and our job is to find the combination that minimizes
  that loss function. The next section is dedicated to procedure that is commonly
  used to find the mimimum loss.
</p>
<div class="separator" />

<style>
  .flex-group {
    display: flex;
    align-items: center;
    justify-items: center;
  }
  .scatter {
    flex: 2;
  }
  .text {
    flex: 1;
  }
</style>
