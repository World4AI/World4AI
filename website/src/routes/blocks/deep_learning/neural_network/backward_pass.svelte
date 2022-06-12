<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Plot from "$lib/Plot.svelte";
  import { NeuralNetwork } from "$lib/NeuralNetwork.js";
  import Button from "$lib/Button.svelte";
  import BackwardPass from "./_backward/BackwardPass.svelte";

  const alpha = 0.5;
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
  <title>World4AI | Deep Learning | Backward Pass</title>
  <meta
    name="description"
    content="Backpropagation is the algorithm that efficiently calculates the gradients of the loss with respect to weights and biases of all the layers in the neural network."
  />
</svelte:head>

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
      >{String.raw`\dfrac{\partial}{\partial w^{<l>}_{k,t}} L(\mathbf{x,b})`}</Latex
    > and <Latex
      >{String.raw`\dfrac{\partial}{\partial b^{<l>}_{k}} L(\mathbf{x,b})`}</Latex
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
  <p>
    The backpropagation algorithm makes extensive use of the chain rule, which
    allows us to find derivatives of composite functions of the form <Latex
      >{String.raw`f(x) = h(g(x))`}</Latex
    >.
  </p>
  <p>
    The chain rule states that to find the derivative of <Latex>f(x)</Latex> with
    respect to <Latex>x</Latex> can be achieved by separately finding the derivatives
    <Latex>{String.raw`\dfrac{dh}{dg}`}</Latex> and <Latex
      >{String.raw`\dfrac{dg}{dx}`}</Latex
    > and to calculate the product of both derivatives.
  </p>
  <Latex
    >{String.raw`
  \dfrac{d}{dx}f(x) = \dfrac{dh}{dg} \dfrac{dg}{dx} \\
    `}</Latex
  >
  <p>
    Intuitively this makes sense because the terms cancel out and we are left
    with our desired derivative.
  </p>
  <Latex
    >{String.raw`
  \dfrac{d}{dx}f(x) = \dfrac{dh}{dg} \dfrac{dg}{dx} = \dfrac{dh}{\cancel{dg}} \dfrac{\cancel{dg}}{dx} = \dfrac{dh}{dx} \\
    `}</Latex
  >
  <p>
    Let us work through a simple example to make the intuition stick. We assume
    we deal with <Latex>{String.raw`f(x) = (5x+100)^2`}</Latex>. The function <Latex
      >f(x)</Latex
    > is actually a composite of the function <Latex
      >{String.raw`h(g) = g^2`}</Latex
    > and <Latex>{String.raw`g(x) = 5x + 100`}</Latex>.
  </p>
  <p>
    First we find the derivatives <Latex
      >{String.raw`\dfrac{dh}{dg} = 2g = 2(5x+100)`}</Latex
    > and <Latex>{String.raw`\dfrac{dg}{dx} = 5`}</Latex>. Lastly we apply the
    chain rule by multiplying the derivatives to end up with <Latex
      >{String.raw`\dfrac{d}{dx}f(x) = 10(5x+100) = 50x + 1000`}</Latex
    >. We can easily verify that this is the correct derivative by calculating
    the derivative directly after applying the binomial formula.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
    \dfrac{d}{dx}f(x) & = \dfrac{d}{dx} (5x + 100)^2 \\
    & = \dfrac{d}{dx}25x^2 + 1000x + 1000 \\
    & = 50x + 1000

  \end{aligned}
    `}</Latex
  >
  <p>
    While we used a composition of two functions for the ease of explanations,
    the chain rule is not restricted to a composition of just two functions. For
    example given a composite function of the following form
    <Latex>{String.raw`f(x) = h(g(u(v(x))))`}</Latex> the derivative can be calculated
    as follows using the chain rule <Latex
      >{String.raw`\dfrac{d}{dx}f(x) = \dfrac{dh}{dg} \dfrac{dg}{du} \dfrac{du}{dv} \dfrac{dv}{dx}`}</Latex
    >.
  </p>
  <p>
    The chain rule is also not restricted to functions with just one variable.
    Let us for example assume that we use two single variable functions
    <Latex>x(t)</Latex> and <Latex>y(t)</Latex>. Let us further assume that we
    calculate the function <Latex>f(x,y)</Latex>. The derivative of <Latex
      >f(x,y)</Latex
    > with respect to <Latex>t</Latex> can be calculated using the multivariable
    chain rule, where
    <Latex
      >{String.raw`\dfrac{d}{dt}f(x, y) = \dfrac{df}{dx} \dfrac{dx}{dt} + \dfrac{df}{dy} \dfrac{dy}{dt}`}</Latex
    >.
  </p>
  <p class="info">
    A neural network is a composition of many different functions with many
    variables.
  </p>
  <p>
    When we look at the neural network from the previous section, we should
    realize, that the network is a composion of many functions with many
    variables.
  </p>
  <Latex
    >{String.raw`\mathbf{\hat{y}} = \sigma( \sigma(\sigma(\mathbf{X} \mathbf{W}^{<1>T}) \mathbf{W}^{<2>T})\mathbf{W}^{<3>T})`}</Latex
  >
  <p>
    That means that we can calculate the gradient of the loss function
    <Latex>{String.raw`L(\mathbf{w}, b)`}</Latex> with respect to any of the layers
    weights <Latex>{String.raw`\mathbf{w}`}</Latex> using the chain rule.
  </p>
  <p>
    But backpropagation is more than just the chain rule, it is the chain rule
    combined with an efficient calculation algorithm.
  </p>
  <p class="info">
    Backpropagation is chain rule plus an efficient algorithm for the
    calculation of the gradient of the loss function with respect to the weights
    and biases.
  </p>
  <p>
    To understand why we need an efficient algorithm, let us use the interactive
    example below. You can click on any of the weights to mark the nodes and
    paths that are necessary to calculate the derivative for that particular
    weight. This is the neural network that we are going to utilize to solve our
    non linear (circular) problem from the start of this chapter.
  </p>
  <BackwardPass />
  <p>
    There are a couple of things that we should notice while playing with the
    example.
  </p>
  <p>
    First, the weights that connect the inputs (leftmost layer) and second layer
    have an impact on all neurons in the third layer. This is the reason why we
    need to use the multivariable chain rule.
  </p>
  <p>
    Second, different weights have an impact on the same neurons. For example
    all the weights that connect the first and the second layer, have an impact
    on all the activations in the third layer and half the weights that connect
    the second and the third layer have an impact on the first activation in the
    third layer and half on the second activation in the third layer. This
    implies that when we apply the chain rule, several weights will have many of
    the same components in the product of the chain rule.
  </p>
  <p>
    To avoid duplicate calculations, some of the results will be saved in the
    backward pass. Backpropagation calculates the gradients for the weights of
    the last layer first. Many of the intermediate results are required parts
    for the calculation of gradients for of all the previous layers. The
    calculations of the gradients in the second to the last layers are also
    saved, as those are required parts for all the layers that precede the
    second to last layer. The process is repeated until the weights that connect
    the inputs with the first hidden layer is reached.
  </p>
  <p>
    It makes sense to work through at least one example to understand how the
    backpropagation algorithm works. For that we are going to use a simple
    neural network, with a two inputs, two hidden layers with two neurons each
    and a single neuron in the output layer.
  </p>
  <BackwardPass sizes={[2, 2, 2, 1]} />
  <p>
    Our loss is defined as below, where <Latex>y</Latex> is the correct class and
    <Latex>{String.raw`\hat{y}`}</Latex> is the prediction from the neural network,
    the probability to belong to the class 1.
  </p>
  <Latex
    >{String.raw`L(\mathbf{w}, b) = -\dfrac{1}{n}\sum_i \big[y^{(i)}\ln \hat{y}^{(i)} + (1-y^{(i)})\ln (1 - \hat{y}^{(i)})\big]`}</Latex
  >
  <p>
    While we always work with a batch or mini batch, for the sake of simplicity
    we are going to focus on a single training example. This simplifies the
    explanations and the notation significantly.
  </p>
  <Latex
    >{String.raw`L(\mathbf{w}, b) = - y\ln \hat{y} -(1-y)\ln(1 - \hat{y}) `}</Latex
  >
  <p>
    Above we change the signs in the calculation, because we put the minus sign
    in front of the sum inside the squared braces.
  </p>
  <p>We continue rewriting the loss function in the following way.</p>
  <Latex
    >{String.raw`L(\mathbf{w}, b) = - y\ln a^{<3>}-(1-y)\ln(1 - a^{<3>}) `}</Latex
  >
  <p>
    We can do that because the last activation <Latex
      >{String.raw`a^{<3>}`}</Latex
    > is the prediction that is produced by the neural network <Latex
      >{String.raw`\hat{y}`}</Latex
    >.
  </p>
  <p>
    First we calculate the derivative of the loss with respect to the last
    activation. We utilize the fact that the derivative of the natural log is <Latex
      >{String.raw`\dfrac{d}{dx}lnx = \dfrac{1}{x}`}</Latex
    > and additionally use the chain rule and end up with.
  </p>
  <Latex
    >{String.raw`\dfrac{d}{da}L = -y\dfrac{1}{a^{<3>}} + (1- y)\dfrac{1}{1-a^{<3>}} `}</Latex
  >
  <p>
    In the next step we calculate the derivative <Latex
      >{String.raw`\dfrac{d}{dz}a(z)`}</Latex
    >. We use the sigmoid as our activation function <Latex
      >{String.raw`\sigma(z) = \dfrac{1}{1+e^{-z}}`}</Latex
    >. While the derivation of the derivative can be somewhat involved, the
    derivative itself is extremely simple, <Latex
      >{String.raw`\dfrac{d}{dz}a(z) = a(1-a)`}</Latex
    >.
  </p>
  <p>
    With those two derivatives in hand we can calculate <Latex
      >{String.raw`\delta^{<3>}`}</Latex
    > as <Latex
      >{String.raw`\delta^{<3>} = \dfrac{dL}{da^{<3>}} \dfrac{da^{<3>}}{dz}`}</Latex
    >. The deltas <Latex>\delta</Latex> are the parts of the calculation that are
    going to be used for the gradients in the previous layers and the current layer.
    The deltas are always partial derivatives with respect to the net inputs <Latex
      >z</Latex
    > of that particular layer.
  </p>
  <p>
    Once we have the <Latex>\delta</Latex> for the last layer, we can reuse this
    value to calculate the gradient with respect to the weights that connect the
    second to last and the last layers.
  </p>
  <Latex
    >{String.raw`\dfrac{dL}{dw^{<3>}_{1,1}} =  \dfrac{dL}{da^{<3>}}\dfrac{da^{<3>}}{dz^{<3>}} \dfrac{dz^{<3>}}{dw_{1,1}^{<3>}} = \delta^{<3>}\dfrac{dz^{<3>}}{dw_{1,1}^{<3>}}`}</Latex
  >
  <br />
  <br />
  <Latex
    >{String.raw`\dfrac{dL}{dw^{<3>}_{1,2}} =  \dfrac{dL}{da^{<3>}}\dfrac{da^{<3>}}{dz^{<3>}} \dfrac{dz^{<3>}}{dw_{1,2}^{<3>}} = \delta^{<3>}\dfrac{dz^{<3>}}{dw_{1,2}^{<3>}}`}</Latex
  >
  <br />
  <br />
  <Latex
    >{String.raw`\dfrac{dL}{db^{<3>}} =  \dfrac{dL}{da^{<3>}}\dfrac{da^{<3>}}{dz^{<3>}} \dfrac{dz^{<3>}}{db^{<3>}} = \delta^{<3>}`}</Latex
  >
  <p>
    Even at this point it should become apparent, that we can reuse the <Latex
      >\delta</Latex
    > to make the computations of the gradients more efficient.
  </p>
  <p>
    After the calculation of the gradients for the weights and biases of the
    last layer, we can move one step backwards. Once again we calculate the
    deltas <Latex>{String.raw`\delta`}</Latex>, but this time we can reuse the <Latex
      >\delta</Latex
    > from the next layer. Let us remember that the deltas <Latex>\delta</Latex>
    are derivatives with respect to the net inputs <Latex>z</Latex>. Therefore
    there are as many deltas as there are neurons in a layer. In the second to
    last layer there are two neurons, so we have to calculate two deltas.
  </p>
  <Latex
    >{String.raw`\delta^{<2>}_1 = \delta^{<3>} \dfrac{dz^{<3>}}{da^{<2>}_1} \dfrac{da^{<2>}_1}{dz_1^{<2>}}`}</Latex
  >
  <br />
  <br />
  <Latex
    >{String.raw`\delta^{<2>}_2 = \delta^{<3>} \dfrac{dz^{<3>}}{da^{<2>}_2} \dfrac{da^{<2>}_2}{dz_2^{<2>}}`}</Latex
  >
  <p>
    These two deltas can be reused several times to calculate the gradients of
    the loss <Latex>L</Latex> with respect to the weights and biases in the second
    layer.
  </p>
  <Latex
    >{String.raw`\dfrac{dL}{dw^{<2>}_{1,1}} = \delta_1^{<2>} \dfrac{dz^{<2>}_1}{dw^{<2>}_{1,1}}`}</Latex
  >
  <br />
  <br />
  <Latex
    >{String.raw`\dfrac{dL}{dw^{<2>}_{1,2}} = \delta_1^{<2>} \dfrac{dz^{<2>}_1}{dw^{<2>}_{1,2}}`}</Latex
  >
  <br />
  <br />
  <Latex>{String.raw`\dfrac{dL}{db^{<2>}_{1}} = \delta_1^{<2>}`}</Latex>
  <br />
  <br />
  <Latex
    >{String.raw`\dfrac{dL}{dw^{<2>}_{2,1}} = \delta_2^{<2>} \dfrac{dz^{<2>}_2}{dw^{<2>}_{2,1}}`}</Latex
  >
  <br />
  <br />
  <Latex
    >{String.raw`\dfrac{dL}{dw^{<2>}_{2,2}} = \delta_2^{<2>} \dfrac{dz^{<2>}_2}{dw^{<2>}_{2,2}}`}</Latex
  >
  <br />
  <br />
  <Latex>{String.raw`\dfrac{dL}{db^{<2>}_{2}} = \delta_2^{<2>}`}</Latex>
  <br />
  <br />
  <p>
    As you might imagine this pattern can theoretically continue indefinetely.
    No matter how many layers the neural network has, we calculate the deltas
    based on the deltas from the next layers and thus avoid inefficient
    recalculations.
  </p>
  <p>
    When we deal with several hidden layers and/or several outputs, we
    inevitably encounter the chain rule in the calculation of the <Latex
      >\delta</Latex
    >. This is due to the fact, that some of the activations and thus net inputs
    are used as the inputs for all the neurons in the next layer.
  </p>
  <Latex
    >{String.raw`\delta^{<1>}_1 = \delta^{<2>}_1 \dfrac{dz^{<2>}_1}{da^{<1>}_1} \dfrac{da^{<1>}_1}{dz_1^{<1>}} + \delta^{<2>}_2 \dfrac{dz^{<2>}_2}{da^{<1>}_1} \dfrac{da^{<1>}_1}{dz_1^{<1>}}`}</Latex
  >
  <br />
  <br />
  <Latex
    >{String.raw`\delta^{<1>}_2 = \delta^{<2>}_1 \dfrac{dz^{<2>}_1}{da^{<1>}_2} \dfrac{da^{<1>}_2}{dz_2^{<1>}} + \delta^{<2>}_2 \dfrac{dz^{<2>}_2}{da^{<1>}_2} \dfrac{da^{<1>}_2}{dz_2^{<1>}}`}</Latex
  >
  <div class="separator" />
  <p>
    Remember that our original goal was to solve the non linear problem of the
    below kind.
  </p>
  <Plot {pointsData} {config} />
  <p>
    In the example below you can observe how the decision boundary moves when
    you use backpropagation. Before you move to that example, we have to warn
    you that you are not dealing with the most efficient implementation. For
    once we use batch and not mini batch gradient descent. That can slow down
    training considerably, because in mini batch gradient descent the algorithm
    takes many gradient descent steps in one epoch and thus moves towards the
    optimal value several times in an epoch, while in simple batch gradient
    descent only one optimization step is taken during an epoch. Additionally we
    initialize the weights randomly using the standard normal distribution and
    the sigmoid activation function. Both are not used often any more, but we
    will deal with improvements for that in a later chapter.
  </p>
  <p>
    Usually 20000 steps are sufficient to find weights for a good decision
    boundary, but the speed depends greatly on the initial weights. Try to
    observe how the cross entropy and the shape of the decision boundary change
    over time. At a certain point you will most likely see a sharp drop in cross
    entropy, this is when things will start to improve greatly.
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
      <Plot
        pathsData={lossData}
        config={{
          width: 500,
          height: 500,
          maxWidth: 600,
          minX: 0,
          maxX: 20000,
          minY: 0,
          maxY: 1,
          xLabel: "Epochs",
          yLabel: "Cross-Entropy Loss",
          padding: { top: 20, right: 40, bottom: 40, left: 60 },
          radius: 5,
          colors: [
            "var(--main-color-1)",
            "var(--main-color-2)",
            "var(--text-color)",
          ],
          xTicks: [
            0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000,
          ],
          yTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }}
      />
    </div>
  </div>
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
