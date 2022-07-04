<script>
  import Container from "$lib/Container.svelte";
  import Plot from "$lib/Plot.svelte";
  import Highlight from "$lib/Highlight.svelte";

  let pointsData1 = [
    { x: 0, y: 0.05 },
    { x: 0.1, y: 0.12 },
    { x: 0.21, y: 0.19 },
    { x: 0.31, y: 0.19 },
    { x: 0.41, y: 0.27 },
    { x: 0.51, y: 0.32 },
    { x: 0.61, y: 0.39 },
    { x: 0.69, y: 0.47 },
    { x: 0.81, y: 0.59 },
    { x: 0.92, y: 0.53 },
    { x: 1, y: 0.62 },
  ];

  let pathsData1 = [
    { x: 0, y: 0.02 },
    { x: 1, y: 0.61 },
  ];

  let pointsData2 = [
    [
      { x: 0, y: 0.05 },
      { x: 0.1, y: 0.12 },
      { x: 0.21, y: 0.19 },
      { x: 0.31, y: 0.19 },
      { x: 0.41, y: 0.27 },
      { x: 0.51, y: 0.32 },
      { x: 0.61, y: 0.39 },
      { x: 0.69, y: 0.47 },
      { x: 0.81, y: 0.59 },
      { x: 0.92, y: 0.53 },
      { x: 1, y: 0.62 },
    ],
    [
      { x: 0.75, y: 0.5 },
      { x: 0.54, y: 0.4 },
      { x: 0.25, y: 0.18 },
      { x: 0.04, y: 0.05 },
    ],
  ];
  let pathsData2 = [
    [
      { x: 0, y: 0.05 },
      { x: 0.1, y: 0.12 },
      { x: 0.21, y: 0.19 },
      { x: 0.31, y: 0.19 },
      { x: 0.41, y: 0.27 },
      { x: 0.51, y: 0.32 },
      { x: 0.61, y: 0.39 },
      { x: 0.69, y: 0.47 },
      { x: 0.81, y: 0.59 },
      { x: 0.92, y: 0.53 },
      { x: 1, y: 0.62 },
    ],
    [
      { x: 0, y: 0.02 },
      { x: 1, y: 0.61 },
    ],
  ];

  let pointsData3 = [];
  let pathsData3 = [];
  for (let i = -3; i <= 3.1; i += 0.05) {
    let noiseMin = -0.8;
    let noiseMax = 0.8;
    let noise = Math.random() * (noiseMax - noiseMin) + noiseMin;
    let x = i;
    let y = i ** 2;
    let noiseY = y + noise;
    pointsData3.push({ x, y: noiseY });
    pathsData3.push({ x, y });
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Overfitting</title>
  <meta
    name="description"
    content="Overfitting and underfitting are two common problems in machine learning. When the model overfits the data, it tracks the training data too closely and does not generalize well to new data. Underfitting on the other hand means that the model is not powerfull enought to fit the data."
  />
</svelte:head>

<h1>Overfitting</h1>
<div class="separator" />
<Container>
  <p>
    Let us assume that we face the data below and we contemplate how we are
    going to fit some model to the data.
  </p>
  <Plot
    pointsData={pointsData1}
    config={{
      xLabel: "Feature",
      yLabel: "Label",
    }}
  />
  <p>
    There seems to be a linear relationship between the feature and the label,
    therefore we decide to use a linear regression model.
  </p>
  <Plot
    pointsData={pointsData1}
    pathsData={pathsData1}
    config={{
      xLabel: "Feature",
      yLabel: "Label",
    }}
  />
  <p>
    We could also train a neural network and would end up with a model that fits
    the data very closely.
  </p>
  <Plot
    pointsData={pointsData1}
    pathsData={pointsData1}
    config={{
      xLabel: "Feature",
      yLabel: "Label",
      pathsColors: ["var(--main-color-1)", "var(--text-color)"],
    }}
  />
  <p>
    While linear regression can only model linear relationships, a neural
    network can assume any imaginable form. When we try to reduce the mean
    squared error, the neural network will inevitably assume a form that fits
    the data as close as possible. Given enough neurons and layers we might end
    up fitting the data perfectly. Yet when you look at both models above you
    might end up thinking that a simple linear model is actually a much better
    fit. The keyword we are looking for is <Highlight>generalization</Highlight
    >.
  </p>
  <p>
    <span class="info"
      >A model that generalizes well is good at modelling new unforseen data.</span
    >
  </p>
  <p>
    When the relationship between the feature and the label turns out to be
    truly linear, the neural network will underperform when new data comes along
    (blue dots). The simpler linear regression model on the other hand will do
    just fine. In other words the linear model generalizes much better in this
    example.
  </p>
  <Plot
    pointsData={pointsData2}
    pathsData={pathsData2}
    config={{
      xLabel: "Feature",
      yLabel: "Label",
      pathsColors: ["var(--main-color-1)", "var(--text-color)"],
    }}
  />
  <p>
    The problem of the model that fits too closely to the training data and does
    not generalize well to previously unforseen data is called <Highlight
      >overfitting</Highlight
    >. Overfitting is a common problem in deep learning and we are going to
    spend this entire section discussing different remedies, starting with ways
    to measure the level of overfitting.
  </p>
  <p>
    As you might imagine <Highlight>underfitting</Highlight> can also be a potential
    problem in deep learning.
  </p>
  <p class="info">
    A model that does not have enough expressiveness or parameters to fit to the
    data will underfit the data.
  </p>
  <p>
    In the example below we face data produced by a quadratic function, yet we
    attempt to use linear regression to fit the data. No matter how hard we try,
    even during training we are going to underperform.
  </p>
  <Plot
    pointsData={pointsData3}
    pathsData={[
      { x: -5, y: 0 },
      { x: 5, y: 11 },
    ]}
    config={{
      xLabel: "Feature",
      yLabel: "Label",
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -5,
      maxX: 5,
      minY: 0,
      maxY: 11,
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 2,
      xTicks: [],
      yTicks: [],
      numTicks: 5,
    }}
  />
  <p>
    Solving underfitting is usually a much easier problem, because it is usually
    sufficient to increase the parameters of the model. We could for example use
    a neural network to train a model that fits better. Increasing the number of
    layers and/or the number of neurons usually solves the problem of
    underfitting.
  </p>
  <Plot
    pointsData={pointsData3}
    pathsData={pathsData3}
    config={{
      xLabel: "Feature",
      yLabel: "Label",
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -5,
      maxX: 5,
      minY: 0,
      maxY: 11,
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 2,
      xTicks: [],
      yTicks: [],
      numTicks: 5,
    }}
  />
  <p>
    In a later chapter we will discuss that we can not increase the number of
    layers indefinetly, even if theoretically more layers have the potential to
    further decrease underfitting.
  </p>
  <div class="separator" />
</Container>
