<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Plot from "$lib/Plot.svelte";

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
  let inputs = [];
  let outputs = [];

  // create the data for the neural network
  function createData() {
    pointsData.forEach((label, labelIdx) => {
      label.forEach((dataPoint) => {
        let input = [];
        input.push(dataPoint.x);
        input.push(dataPoint.y);
        inputs.push(input);
        outputs.push(labelIdx);
      });
    });
  }

  createData();

  // learnable parameters of the neural network
  let weights = [
    [
      [-0.2, -0.3],
      [-0.5, -0.5],
      [-0.5, -0.1],
      [-0.2, -0.3],
    ],
    [
      [-1, -0.2, -0.1, -1],
      [-0.2, -0.1, -0.3, 0.2],
    ],
    [[0.45, 1]],
  ];
  let biases = [[0, -0.5, 0, -0.5], [-1, -0.5], [-0.4]];

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  //use the weights, biases and sigmoid to calculate the output for a single sample
  function feedForward(input) {
    let outputs = [[...input]];
    weights.forEach((layer, layerIdx) => {
      let layerOutput = [];
      layer.forEach((nodeWeights, nodeIdx) => {
        let zValue = 0;
        nodeWeights.forEach((weight, weightIdx) => {
          zValue += weight * outputs[layerIdx][weightIdx];
        });
        zValue += biases[layerIdx][nodeIdx];
        let activation = sigmoid(zValue);
        layerOutput.push(activation);
      });
      outputs.push(layerOutput);
    });
    return outputs;
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
    heatmapCoordinates.forEach((inputs) => {
      let outputs = feedForward(inputs);
      let prediction = outputs[outputs.length - 1];
      let label;
      if (prediction[0] >= 0.5) {
        label = 1;
      } else {
        label = 0;
      }
      let point = { x: inputs[0], y: inputs[1], class: label };
      heatmapData.push(point);
    });
  }

  calculateHeatmap();

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
</script>

<h1>Backward Pass</h1>
<div class="separator" />
<Container>
  <p>
    In the forward pass we calculate the label estimates <Latex
      >{String.raw`\mathbf{\hat{y}}`}</Latex
    > based on the features <Latex>{String.raw`\mathbf{X}`}</Latex>. These
    estimates allow us to measure the loss <Latex
      >{String.raw`L(\mathbf{w,b})`}</Latex
    >, like the cross-entropy loss for classification or the mean squared error
    for regression.
  </p>
  <p>
    In the backward pass we calculate the gradients and apply gradient descent.
  </p>
  <Latex
    >{String.raw` \mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}\\ `}</Latex
  >
  <Latex
    >{String.raw` \mathbf{b}_{t+1} \coloneqq \mathbf{b}_t - \alpha \mathbf{\nabla}\\ `}</Latex
  >
  <p>
    This procedure should look familiar, because in essence these are the same
    steps, that we used in linear and logistic regression, but how exactly
    should we calculate those gradients? When we define gradient descent as <Latex
      >{String.raw` \mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}`}</Latex
    >, we imply that the vector <Latex>{String.raw`\mathbf{w}`}</Latex> encompasses
    all the weights from all the layers in the neural network. Consider for example
    the forward pass from the previous section.
  </p>
  <Latex
    >{String.raw`\mathbf{\hat{y}} = \sigma( \sigma(\sigma(\mathbf{X} \mathbf{W}^{<1>T}) \mathbf{W}^{<2>T})\mathbf{W}^{<3>T})`}</Latex
  >
  <p>
    Just two small hidden layers and an output layer create a complex nested
    expression and yet we are expected to exactly calculate all the derivatives
    of the form <Latex
      >{String.raw`\dfrac{\delta}{\delta w^{<l>}_{k,t}} L(\mathbf{x,b})`}</Latex
    > and <Latex
      >{String.raw`\dfrac{\delta}{\delta b^{<l>}_{k}} L(\mathbf{x,b})`}</Latex
    >, where the superscript <Latex>{String.raw`<l>`}</Latex> is the layer in the
    neural network, <Latex>k</Latex> is the row in the weight matrix and <Latex
      >t</Latex
    > is the column in the weight matrix. In other words we need to figure out how
    individual weights and biases have an impact on the loss in order to adjust those
    weights and biases to reduce the loss. That is exactly what the <Highlight
      >backpropagation algorithm</Highlight
    > is used for.
  </p>
  <p class="info">
    The backpropagation algorithm calculates the gradients of weights and biases
    for each layer of the neural network.
  </p>
  <Plot {pointsData} {config} />
  <Plot {pointsData} {heatmapData} {config} />
</Container>
