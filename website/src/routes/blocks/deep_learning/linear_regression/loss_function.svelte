<script>
  import Container from "$lib/Container.svelte";
  import Plot from "$lib/Plot.svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";
  import Mse from "./_loss/Mse.svelte";

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

  let staticRegressionLine = [];
  let staticLines = [];

  function staticData() {
    //regression line
    let x1 = 0;
    let y1 = b + w * x1;
    let x2 = 60;
    let y2 = b + w * x2;
    let line = [
      { x: x1, y: y1 },
      { x: x2, y: y2 },
    ];

    staticRegressionLine.push(line);
    staticLines.push(line);

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
      staticLines.push(line);
    });
  }

  staticData();
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Linear Regression Loss</title>
  <meta
    name="description"
    content="Linear regression utilizes the mean squared error as a measure of succuss or failure. The goal is to minimize the MSE."
  />
</svelte:head>

<Container>
  <h1>Loss Function</h1>
  <div class="separator" />
  <p>
    While we can intuitively tell how far away our line is from some optimal
    location, we need a quantitative measure that the linear regression
    algorithm can use for optimization purposes. In machine learning we use a
    measure called loss (or loss function). Using this quantity we can tweak the
    weights and biases to find a minimum value of the loss.
  </p>
  <p>
    In our example we will use only 4 data points. A bigger dataset would
    otherwise clutter the illustrations.
  </p>
  <Plot
    pointsData={data}
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
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
      xTicks: [],
      yTicks: [],
      numTicks: 7,
    }}
  />
  <p>
    We will start by drawing a 45 degree line from the (0,0) to the (60, 60)
    position. While this looks "OK", we do net have a way to compare that
    particular line with any other lines.
  </p>
  <Plot
    pointsData={data}
    pathsData={staticRegressionLine}
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

  <p>
    The first step is to calculate the error between the actual label <Latex
      >y_i</Latex
    > and the predicted value <Latex>{String.raw`\hat{y}_i=b+wx_i`}</Latex> for each
    datapoint
    <Latex>i</Latex>. We can define the difference <Latex
      >{String.raw`y_i - \hat{y}_i`}</Latex
    > as the error. Visually we can draw that error as the line that connects the
    regression line with the true label.
  </p>
  <Plot
    pointsData={data}
    pathsData={staticLines}
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
  <p>
    If we tried to sum up all the errors in the dataset <Latex
      >{String.raw`\sum_i^n \hat{y}_i - y_i`}</Latex
    > we would realize that the positive and the negative errors are offsetting each
    other. If the errors were symmetrical above and below the line, we would end
    up with a summed error of 0.
  </p>
  <p>
    The loss that used in linear regression, the mean squared error (MSE), takes
    the error to the power of 2 to get rid of the negative sign. The error is
    then divided by the number of data points to calculate the mean (average) of
    the error. The mean squared error is mathematically expressed as follows.
  </p>
  <Latex>{String.raw`\Large MSE=\frac{1}{n}\sum_i^n (\hat{y} - y_i)^2`}</Latex>

  <p>
    Visually it actually makes sense to imagine squares. Each data point has a
    corresponding square and the larger the area of that square, the larger the
    contribution to the MSE. The length of the side of the square is calculated
    as the absolute distance between <Latex>y</Latex> and <Latex
      >{String.raw`\hat{y}`}</Latex
    >. Try to use the example below and move the weight and the bias. Observe
    how the mean squared error changes based on the parameters.
  </p>

  <Mse {data} {w} {b} />
  <div class="flex-container">
    <div><Latex>w</Latex></div>
    <Slider bind:value={w} min={-20} max={20} step={0.1} />
  </div>
  <div class="flex-container">
    <div><Latex>b</Latex></div>
    <Slider bind:value={b} min={-50} max={50} />
  </div>
  <p>
    Different combination of the weight <Latex>w</Latex> and the bias <Latex
      >b</Latex
    > produce different losses and our job is to find the combination that minimizes
    that loss function. The next section is dedicated to procedure that is commonly
    used to find the mimimum loss.
  </p>
  <div class="separator" />
</Container>

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
