<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Plot from "$lib/Plot.svelte";
  import { NeuralNetwork } from "$lib/NeuralNetwork.js";

  const alpha = 0.01;
  const sizes = [2, 4, 2, 1];

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
