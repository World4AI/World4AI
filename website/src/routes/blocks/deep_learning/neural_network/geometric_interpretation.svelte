<script>
  import Container from "$lib/Container.svelte";
  import { NeuralNetwork } from "$lib/NeuralNetwork.js";
  import Plot from "$lib/Plot.svelte";
  import Button from "$lib/Button.svelte";
  import Architecture from "./_geometric/Architecture.svelte";
  const alpha = 0.5;
  const sizes = [2, 4, 2, 1];

  //good starting weights for faster convergence
  let weights = [
    [
      [-0.44947513937950134, -2.1187565326690674],
      [0.24469861388206482, 1.4741417169570923],
      [-0.8196889758110046, 1.3501536846160889],
      [0.15400901436805725, -0.35472017526626587],
    ],
    [
      [
        -1.3636902570724487, -1.169247031211853, 0.29788315296173096,
        -1.699813961982727,
      ],
      [
        -1.0098611116409302, 1.1693042516708374, -0.011132504791021347,
        -0.6532079577445984,
      ],
    ],
    [[0.8470180034637451, -1.2319238185882568]],
  ];

  let biases = [
    [
      [
        0.6538878083229065, -1.1869943141937256, -1.317667841911316,
        0.8878940939903259,
      ],
    ],
    [[1.6244404315948486, -0.7040465474128723]],
    [[-0.12584348022937775]],
  ];

  // create the data to draw the svg
  let pointsData = [[], []];
  let radius = [0.45, 0.25];
  let centerX = 0.5;
  let centerY = 0.5;
  for (let i = 0; i < radius.length; i++) {
    for (let point = 0; point < 200; point++) {
      let angle = 2 * Math.PI * Math.random();
      let r = radius[i];
      let x = r * Math.cos(angle) + centerX;
      let y = r * Math.sin(angle) + centerY;
      pointsData[i].push({ x, y });
    }
  }

  //these are the X and the y values
  let features = [];
  let labels = [];

  // create the data for the neural network
  function createData() {
    pointsData.forEach((label, labelIdx) => {
      label.forEach((dataPoint) => {
        let feature = [];
        feature.push(dataPoint.x);
        feature.push(dataPoint.y);
        features.push(feature);
        let label = [];
        label.push(labelIdx);
        labels.push(label);
      });
    });
  }
  createData();

  let nn = new NeuralNetwork(alpha, sizes, features, labels);
  nn.setWeights(weights);
  nn.setBiases(biases);

  const activationsStore = nn.activationsStore;

  let pointsData2 = [[], []];
  $: {
    pointsData2 = [[], []];
    try {
      $activationsStore[$activationsStore.length - 2].forEach((output, idx) => {
        let x = output[0];
        let y = output[1];
        let labelClass = labels[idx][0];
        pointsData2[labelClass].push({ x, y });
      });
    } catch {}
  }

  // determine the x and y coordinates that are going to be used for heatmap
  let numbers = 50;
  let heatmapCoordinates = [];
  for (let i = 0; i < numbers; i++) {
    for (let j = 0; j < numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let coordinate = [];
      coordinate.push(x);
      coordinate.push(y);
      heatmapCoordinates.push(coordinate);
    }
  }

  let heatmapData = [];
  //recalculate the heatmap based on the current weights of the neural network
  function calculateHeatmap() {
    heatmapData = [];
    let outputs = nn.predict(heatmapCoordinates);
    heatmapCoordinates.forEach((inputs, idx) => {
      let label;
      if (outputs[idx] >= 0.5) {
        label = 1;
      } else {
        label = 0;
      }
      let point = { x: inputs[0], y: inputs[1], class: label };
      heatmapData.push(point);
    });
  }

  calculateHeatmap();

  //generate graphs
  let lossStore = nn.lossStore;
  let lossData = [];
  $: {
    let losses = $lossStore;
    let idx = losses.length - 1;
    let loss = losses[idx];
    if (idx !== -1) {
      let point = { x: idx, y: loss };
      lossData.push(point);
      lossData = lossData;
    }
  }
  let config = {
    width: 500,
    height: 500,
    maxWidth: 600,
    minX: 0,
    maxX: 1,
    minY: 0,
    maxY: 1,
    xLabel: "Feature 1",
    yLabel: "Feature 2",
    padding: { top: 20, right: 40, bottom: 40, left: 60 },
    radius: 5,
    colors: ["var(--main-color-1)", "var(--main-color-2)", "var(--text-color)"],
    heatmapColors: ["var(--main-color-3)", "var(--main-color-4)"],
  };

  let runImprovements = false;
  let runs = 0;

  function runEpoch() {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        nn.epoch();
        calculateHeatmap();
        resolve();
      }, 0);
    });
  }

  async function train() {
    runImprovements = true;
    while (runImprovements) {
      await runEpoch();
      runs++;
    }
  }

  function stopTraining() {
    runImprovements = false;
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Geometric Interpretation</title>
  <meta
    name="description"
    content="Neural networks transform the inputs by scaling, translating and rotating the data with matrix multiplication. Additionally matrix multiplications can move the hidden features between different dimensions to better solve the task. Activation functions squish the data to provide solutions for non linear problesm. The last layer is linearly separable."
  />
</svelte:head>

<h1>Neural Networks Geometric Interpretation</h1>
<div class="separator" />

<Container>
  <p>
    Neural networks are basically a collection of linear transformations through
    matrix multiplications and non linear transformations through the
    application of activation functions. But what exactly does that mean when it
    comes to finding the solution to our problem?
  </p>
  <p>
    The architecture that we picked to solve our circular classification problem
    was not picked randomly, but to show some magic that is hidden under the
    hood of a neural network.
  </p>
  <Architecture {sizes} />
  <p>
    Let us remember that logistic regression is able to deal with classification
    problems, but only if the data is linearly separable. The last layer of the
    above neural network looks exacly like a logistic regression. The neuron
    receives two inputs and applies the sigmoid to scaled inputs to produce
    values between 0 and 1. The big difference is that the inputs into the
    logistic regression are the outputs of the last hidden layer. That must
    mean, that the neural network is somehow able to extract features through
    those linear and non linear transformations, that are linearly separable.
  </p>
  <p>
    The example below shows how the neural network learns those transformations.
    On one side you can see the original inputs, on the other side are the two
    extracted features that are used as inputs into the output layer. When the
    neural network has learned to separate the two circles, that means that the
    two features from the last hidden layer are linearly separable. Start the
    example and observe the learning process. At the beginning the hidden
    features are clustered together, but after a while you will notice that you
    could separate the different colors by a single line.
  </p>
</Container>
<div class="separator" />
<Container maxWidth="300px">
  {#if runImprovements}
    <Button value="PAUSE TRAINING" on:click={stopTraining} />
  {:else}
    <Button value="START TRAINING" on:click={train} />
  {/if}
  <span><strong>Epoch: {lossData.length}</strong></span>
</Container>
<div class="separator" />

<Container maxWidth="1900px">
  <div class="flex-container">
    <div class="left-container">
      <Plot {pointsData} {heatmapData} {config} />
    </div>
    <div class="right-container">
      <Plot pointsData={pointsData2} {config} />
    </div>
  </div>
</Container>
<Container>
  <p>
    It is not always clear how the neural network does those transformations,
    but we could use the example above to get some intuition for the process. If
    you look at the original data again you might notice something peculiar.
    Imagine the data is actually located in 3d space and you are looking at the
    data from above. Now imagine that the blue and the red dots are located on
    different heights (z-axis). Wouldn't that mean that you could construct a 2d
    plane in 3d space to linearly separate the data? Yes it would. The first
    hidden layer of our neural network transforms the 2d data into a 4d data.
    While we can not imagine 4d space, we should be able to reason that it is
    easier to separate the circular data in a higher dimension. Afterwards we
    move the processed features back into 2d space, where it is linearly
    separable.
  </p>
  <p>
    Modern neural networks have hundreds or thousands of dimensions and hidden
    layers and we can not visualize the hidden features to get a better feel for
    what the neural network does. But generally speaking the matrix
    multiplications move, scale, rotate the data and move it between different
    dimensions. The activations squish or restraint the data to deal with non
    linearity. The last layers contain the hidden features, that can be linearly
    separated to solve a particular problem.
  </p>
  <div class="separator" />
</Container>

<style>
  .flex-container {
    display: flex;
    flex-direction: row;
  }

  .left-container {
    flex-basis: 1900px;
  }
  .right-container {
    flex-basis: 1900px;
  }

  @media (max-width: 1000px) {
    .flex-container {
      flex-direction: column;
    }
    .left-container {
      flex-basis: initial;
    }
    .right-container {
      flex-basis: initial;
    }
  }
</style>
