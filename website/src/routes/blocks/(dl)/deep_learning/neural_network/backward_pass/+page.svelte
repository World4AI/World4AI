<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  
  import { NeuralNetwork } from "$lib/NeuralNetwork.js";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import BackwardPass from "../_backward/BackwardPass.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte"; 
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import XLabel from "$lib/plt/XLabel.svelte"; 
  import YLabel from "$lib/plt/YLabel.svelte"; 
  import Circle from "$lib/plt/Circle.svelte"; 
  import Rectangle from "$lib/plt/Rectangle.svelte"; 
  import Path from "$lib/plt/Path.svelte"; 

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

  let heatmapData = [[], []];
  //recalculate the heatmap based on the current weights of the neural network
  function calculateHeatmap() {
    heatmapData = [[], []];
    let outputs = nn.predict(heatmapCoordinates);
    heatmapCoordinates.forEach((inputs, idx) => {
      let point = { x: inputs[0], y: inputs[1] };
      if (outputs[idx] >= 0.5) {
        heatmapData[0].push(point);
      } else {
        heatmapData[1].push(point);
      }
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

  let runs = 0;

  function train() {
    runs++;
    nn.epoch();
    calculateHeatmap();
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
    estimates allow us to measure the loss <Latex>{String.raw`L`}</Latex>, like
    the cross-entropy loss for classification or the mean squared error for
    regression.
  </p>
  <p>
    In the backward pass we calculate the gradients and apply gradient descent.
  </p>
  <Latex
    >{String.raw`
    \begin{aligned}
    \mathbf{w}_{t+1} & \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w \\
    \mathbf{b}_{t+1} & \coloneqq \mathbf{b}_t - \alpha \mathbf{\nabla_b}\\ 
    \end{aligned}
    `}
  </Latex>
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
      >{String.raw`\dfrac{\partial}{\partial w^{<l>}_{k,t}} L`}</Latex
    > and <Latex>{String.raw`\dfrac{\partial}{\partial b^{<l>}_{k}} L`}</Latex>,
    where the superscript <Latex>{String.raw`<l>`}</Latex> is the layer in the neural
    network, <Latex>k</Latex> is the row in the weight matrix and <Latex
      >t</Latex
    > is the column in the weight matrix. In other words we need to figure out how
    individual weights and biases impact the loss in order to adjust those weights
    and biases to reduce the loss. That is exactly what the <Highlight
      >backpropagation algorithm</Highlight
    > is used for.
  </p>
  <p class="warning">
    The backpropagation algorithm calculates the gradients of the loss with
    respect to weights and biases for each layer and neuron of the neural
    network. Gradient descent on the other hand utilizes those gradients to
    adjust the weights and biases to reduce the loss.
  </p>
  <p>
    The backpropagation algorithm makes extensive use of the chain rule, which
    allows us to find derivatives of composite functions of the form <Latex
      >{String.raw`f(x) = h(g(x))`}</Latex
    >.
  </p>
  <p>
    The chain rule states that we can find the derivative of <Latex>f(x)</Latex>
    with respect to <Latex>x</Latex> by separately finding the derivatives
    <Latex>{String.raw`\dfrac{dh}{dg}`}</Latex> and <Latex
      >{String.raw`\dfrac{dg}{dx}`}</Latex
    > calculating the product of both derivatives.
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
    as follows using the chain rule: <Latex
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
  <p>
    When we look at the neural network from the previous section, it becomes
    clear, <span class="light-blue"
      >that the network is a composion of many functions with many variables<span
        >.
      </span></span
    >
  </p>
  <Latex
    >{String.raw`\mathbf{\hat{y}} = \sigma( \sigma(\sigma(\mathbf{X} \mathbf{W}^{<1>T}) \mathbf{W}^{<2>T})\mathbf{W}^{<3>T})`}</Latex
  >
  <p>
    That means that we should be able to calculate the partial derivative of the
    loss function
    <Latex>{String.raw`L`}</Latex> with respect to any of the weights <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > or biases <Latex>{String.raw`\mathbf{b}`}</Latex> using the chain rule.
  </p>
  <p>
    But backpropagation is more than just the chain rule, <span
      class="light-blue"
      >it is the chain rule combined with an efficient calculation algorithm.</span
    >
  </p>
  <p>
    To understand why we need an efficient algorithm, let us use the interactive
    example below. You can click on any of the weights to mark the nodes and
    paths that are necessary to calculate the derivative for that particular
    weight.
  </p>
  <BackwardPass />
  <p>
    The weights that connect the inputs (leftmost layer) and the second layer
    have an impact on all neurons in the third layer. This is essentially the
    reason why we need to use the multivariable chain rule.
  </p>
  <p>
    Neurons are influenced by many different weights from previous layers. For
    example all the weights that connect the first and the second layer, have an
    impact on the first activation in the third layer and half the weights that
    connect the second and the third layer have an impact on the first
    activation in the third layer. This implies that when we apply the chain
    rule, gradients of several weights will have many of the same components in
    the product of the chain rule. To avoid duplicate calculations, some of the
    results will be saved in the backward pass. Backpropagation calculates the
    gradients for the weights of the last layer first. Many of the intermediate
    results are required parts for the calculation of gradients for of all the
    previous layers. The calculations of the gradients in the second to the last
    layers are also saved, as those are required parts for all the layers that
    precede the second to last layer. The process can be applied indefinetely,
    no matter how many layers the neural network has.
  </p>
  <p>
    It makes sense to work through at least one example to understand how the
    backpropagation algorithm works. For that we are going to use a simple
    neural network. The network has two features as inputs, two hidden layers
    with two neurons each and the output layer has a single neuron.
  </p>
  <BackwardPass sizes={[2, 2, 2, 1]} />
  <p>
    In this example we utilize the cross-entropy, but the similar derivations
    can be made with the mean squared error loss.
  </p>
  <Latex
    >{String.raw`L = -\dfrac{1}{n}\sum_i \big[y^{(i)}\ln \hat{y}^{(i)} + (1-y^{(i)})\ln (1 - \hat{y}^{(i)})\big]`}</Latex
  >
  <p>
    While we always work with a batch or mini batch in practice, for the sake of
    simplicity we are going to focus on a single training example. This
    simplifies the explanations and the notation significantly. Just keep in
    mind that in practice we would make the calculations for a whole batch and
    calculate the mean of the gradients, before we take a gradient descent step.
  </p>
  <Latex>{String.raw`L = - y\ln \hat{y} -(1-y)\ln(1 - \hat{y}) `}</Latex>
  <p>
    Additionally we replace the prediction output <Latex
      >{String.raw`\hat{y}`}</Latex
    > by the output of the last layer <Latex>{String.raw`a^{<3>}`}</Latex>. This
    is a notational change, that has no impact on the calculation.
  </p>
  <Latex>{String.raw`L = - y\ln a^{<3>}-(1-y)\ln(1 - a^{<3>}) `}</Latex>
  <p>
    We are going to apply the same strategy that we used in linear and logistic
    regression. We keep taking partial derivatives with respect to some
    intermediary values and apply the chain rule once we have all the necessary
    components.
  </p>
  <p>
    First we calculate the derivative of the loss <Latex>L</Latex> with respect to
    the last activation. We utilize the fact that the derivative of the natural log
    is <Latex>{String.raw`\dfrac{d}{dx}lnx = \dfrac{1}{x}`}</Latex>
    and end up with.
  </p>
  <Latex
    >{String.raw`\dfrac{d}{d a^{<3>}}L = -y\dfrac{1}{a^{<3>}} + (1- y)\dfrac{1}{1-a^{<3>}} `}</Latex
  >
  <p>
    In the next step we calculate the derivative <Latex
      >{String.raw`\dfrac{da^{<3>}}{dz^{<3>}}`}</Latex
    >. We use the sigmoid as our activation function <Latex
      >{String.raw`\sigma(z) = \dfrac{1}{1+e^{-z}}`}</Latex
    >, which means that the derivative is
    <Latex>{String.raw`\dfrac{da^{<3>}}{dz^{<3>}} = a^{<3>}(1-a^{<3>})`}</Latex
    >.
  </p>
  <p>
    With those two derivatives in hand we can calculate an intermediary error <Latex
      >{String.raw`\delta^{<3>}`}</Latex
    >. The error is simply the application of the chain rule, therefore <Latex
      >{String.raw`\delta^{<3>} = \dfrac{dL}{da^{<3>}} \dfrac{da^{<3>}}{dz^{<3>}}`}</Latex
    >. The deltas <Latex>\delta</Latex> are the parts of the calculation that are
    going to be used for the gradients in all the previous layers and the current
    layer. The deltas are always partial derivatives with respect to the net inputs
    <Latex>z</Latex> of that particular layer.
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
    >{String.raw`\dfrac{dL}{db^{<3>}} =  \dfrac{dL}{da^{<3>}}\dfrac{da^{<3>}}{dz^{<3>}} \dfrac{dz^{<3>}}{db^{<3>}} = \delta^{<3>}\dfrac{dz^{<3>}}{db^{<3>}}`}</Latex
  >
  <p>
    The net input in the third layer is represented as <Latex
      >{String.raw`z^{<3>}_1 = a^{<2>}_1 w_{11}^{<3>} + a^{<2>}_{12} w_{12}^{<3>} + b_1^{<3>}`}</Latex
    >. Calculating the partial derivative of the net input with respect to
    weights and biases shoud not pose any problems, as this is essentially the
    same exercise we did with linear regression:
    <Latex
      >{String.raw`\dfrac{\partial z^{<3>}_1}{\partial w_{1,1}^{<3>}} = a^{<2>}_1`}</Latex
    >
    ,
    <Latex
      >{String.raw`\dfrac{\partial z^{<3>}_1}{\partial w_{1,2}^{<3>}} = a^{<2>}_2`}</Latex
    >,

    <Latex
      >{String.raw`\dfrac{\partial z^{<3>}_1}{\partial b_1^{<3>}} = 1`}</Latex
    >.
  </p>
  <p>
    After the calculation of the gradients for the weights and biases of the
    last layer, we can move one step backwards. Once again our goal is to
    calculate the deltas <Latex>{String.raw`\delta`}</Latex>, but this time we
    can reuse the <Latex>\delta</Latex> from the next layer. Let us remember that
    the deltas <Latex>\delta</Latex>
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
    This time around we need calculate the derivatives of net input <Latex
      >{String.raw`z^{<3>}_1 = a^{<2>}_1 w_{11}^{<3>} + a^{<2>}_{12} w_{12}^{<3>} + b_1^{<3>}`}</Latex
    > with respect to the activations <Latex>a</Latex>, which turn out to be the
    corresponding weights:
    <Latex
      >{String.raw`\dfrac{\partial z^{<3>}_1}{\partial a^{<2>}_1 } = w_{1,1}^{<3>}`}</Latex
    >
    and
    <Latex
      >{String.raw`\dfrac{\partial z^{<3>}_1}{\partial a^{<2>}_2} = w_{1,2}^{<3>}`}</Latex
    >.
  </p>
  <p>
    At this point we used the delta <Latex>{String.raw`\delta^{<3>}`}</Latex> five
    times. The more layers and neurons we have, the more computational resources
    we save by reusing the deltas. This is the magic of backpropagation.
  </p>
  <p>
    We continue with our path applying the chain rule and calculate the
    gradients with respect to weights and biases in the second layer.
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
    As you might imagine this pattern of computing and reusing deltas can
    theoretically continue indefinetely. No matter how many layers the neural
    network has, we calculate the deltas based on the deltas from the next
    layers and thus avoid inefficient recalculations.
  </p>
  <p>
    When we deal with several hidden layers and/or several outputs, we
    inevitably encounter the multivariable chain rule in the calculation of the <Latex
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
  <p>
    Once we arrive at the first layer we have collected all the necessary
    gradients and we apply gradient descent.
  </p>
  <div class="separator" />
  <p>
    Remember that our original goal was to solve the non linear problem of the
    below kind.
  </p>
   <Plot width={500} height={500} maxWidth={600} domain={[0, 1]} range={[0, 1]}>
     <Ticks xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
            yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
            xOffset={-15} 
            yOffset={15}/>
     <Circle data={pointsData[0]} />
     <Circle data={pointsData[1]} color="var(--main-color-2)" />
     <XLabel text="Feature 1" fontSize={15} />
     <YLabel text="Feature 2" fontSize={15} />
   </Plot>
  <p>
    In the example below you can observe how the decision boundary moves when
    you use backpropagation. Before you move to that example, we have to warn
    you that you are not dealing with the most efficient implementation. For
    once we use batch and not mini batch gradient descent. That can slow down
    training considerably, because in mini batch gradient descent the algorithm
    takes many gradient descent steps in one epoch and thus moves towards the
    optimal value several times in an epoch, while in simple batch gradient
    descent only one optimization step is taken during an epoch. Additionally we
    use the sigmoid activation function, which is not considered to be state of
    the art any more. We want to focus on the most straightforward
    implementation for the time being and introduce improvements in a dedicated
    chapter.
  </p>
  <p>
    Usually 10000 steps are sufficient to find weights for a good decision
    boundary. Try to observe how the cross-entropy and the shape of the decision
    boundary change over time. At a certain point you will most likely see a
    sharp drop in cross entropy, this is when things will start to improve
    greatly.
  </p>
</Container>
<Container maxWidth="1900px">
  <ButtonContainer>
    <PlayButton f={train} delta={0} />
  </ButtonContainer>
  <div class="flex-container">
    <div class="left-container">
      <Plot width={500} height={500} maxWidth={600} domain={[0, 1]} range={[0, 1]}>
        <Ticks xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
               yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
               xOffset={-15} 
               yOffset={15}/>
        <Rectangle data={heatmapData[0]} size={9} color="var(--main-color-3)" />
        <Rectangle data={heatmapData[1]} size={9} color="var(--main-color-4)" />
        <Circle data={pointsData[0]} />
        <Circle data={pointsData[1]} color="var(--main-color-2)" />
        <XLabel text="Feature 1" fontSize={15} />
        <YLabel text="Feature 2" fontSize={15} />
      </Plot>
    </div>
    <div class="right-container">
      <Plot width={500} height={500} maxWidth={600} domain={[0, 10000]} range={[0, 1]}>
        <Ticks xTicks={[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]} 
               yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
               xOffset={-15} 
               yOffset={15}/>
        <XLabel text="Feature 1" fontSize={15} />
        <YLabel text="Feature 2" fontSize={15} />
        <Path data={lossData} />
      </Plot>
    </div>
  </div>
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
