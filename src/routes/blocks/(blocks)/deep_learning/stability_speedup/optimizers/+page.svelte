<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import StepButton from "$lib/button/StepButton.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Contour from "$lib/plt/Contour.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ellipse from "$lib/plt/Ellipse.svelte";
  import Legend from "$lib/plt/Legend.svelte";
  import * as d3 from "d3";

  const references = [
    {
      author: "Diederik Kingma and Jimmy Ba",
      title: "Adam: A Method for Stochastic Optimization",
      journal: "",
      year: "2014",
      pages: "",
      volume: "",
      issue: "",
    },
  ];

  const notes = [
    "RMSProp was developed by Geoffrey Hinton for a deep learning course on the Coursera plattform. You can access the original materials at https://www.cs.toronto.edu/~hinton/. Lecture 6 is the relevant one.",
  ];

  //parameters for function with local minimum
  let localPoint = [];
  let localX = 7;
  let localMomentumX = 7;
  let localAlpha = 0.01;
  let localBeta = 0.9;

  $: localY = localX ** 3 - 5 * localX ** 2 + 10;
  $: localMomentumY = localMomentumX ** 3 - 5 * localMomentumX ** 2 + 10;
  // draw x^3 - 5x^2 + 10 to show local minimum
  let localMinimumData = [];
  for (let i = -6; i <= 7; i += 0.1) {
    let x = i;
    let y = x ** 3 - 5 * x ** 2 + 10;
    localMinimumData.push({ x, y });
  }

  function recalculateLocalMinimumPoints() {
    localPoint = [];
    localPoint.push({ x: localX, y: localY });
    localPoint.push({ x: localMomentumX, y: localMomentumY });
  }

  function localVanillaGradientDescent() {
    // we are dealing with x^3 - 5x^2 + 10
    // the derivative is 3x^2 - 10x
    let gradient = 3 * localX ** 2 - 10 * localX;
    localX = localX - localAlpha * gradient;
  }

  let momentum = null;
  function localMomentumGradientDescent() {
    // we are dealing with x^3 - 5x^2 + 10
    // the derivative is 3x^2 - 10x
    let gradient = 3 * localMomentumX ** 2 - 10 * localMomentumX;
    if (momentum === null) {
      momentum = gradient;
    } else {
      momentum = momentum * localBeta + gradient * (1 - localBeta);
    }
    localMomentumX = localMomentumX - localAlpha * momentum;
  }

  function localGradientDescent() {
    localVanillaGradientDescent();
    localMomentumGradientDescent();
  }

  $: (localY || localMomentumY) && recalculateLocalMinimumPoints();

  /*-------------------------------------------------------*/
  function calculatePath(
    epochs,
    f,
    // grad example (x, y) => {return { x: 2 * x, y: 2 * y };};
    grad,
    startX,
    startY,
    alpha,
    //adaptive -> rmsprop, momentum + adaptive -> adam
    momentum = false,
    adaptive = false,
    beta1 = 0.9,
    beta2 = 0.9,
    // numerical stability rmsprop and adam
    epsilon = 0.0001
  ) {
    let coordinates = [];
    coordinates.push({ x: startX, y: startY });

    let x;
    let y;
    let gradX;
    let gradY;
    let momentumWeighted;
    let adaptiveWeighted;
    if (momentum) {
      momentumWeighted = { x: 0, y: 0 };
    }
    if (adaptive) {
      adaptiveWeighted = { x: 0, y: 0 };
    }

    for (let i = 0; i < epochs; i++) {
      x = coordinates[i].x;
      y = coordinates[i].y;

      gradX = grad(x, y).x;
      gradY = grad(x, y).y;

      let numeratorX = gradX;
      let numeratorY = gradY;

      if (momentum) {
        numeratorX = momentumWeighted.x =
          momentumWeighted.x * beta1 + gradX * (1 - beta1);
        numeratorY = momentumWeighted.y =
          momentumWeighted.y * beta1 + gradY * (1 - beta1);

        let bias = 1 / (1 - beta1 ** (i + 1));
        numeratorX *= bias;
        numeratorY *= bias;
      }

      let denominatorX = 1;
      let denominatorY = 1;
      if (adaptive) {
        denominatorX = adaptiveWeighted.x =
          adaptiveWeighted.x * beta2 + gradX ** 2 * (1 - beta2);
        denominatorY = adaptiveWeighted.y =
          adaptiveWeighted.y * beta2 + gradY ** 2 * (1 - beta2);

        let bias = 1 / (1 - beta2 ** (i + 1));
        denominatorX *= bias;
        denominatorY *= bias;

        denominatorX = Math.sqrt(denominatorX) + epsilon;
        denominatorY = Math.sqrt(denominatorY) + epsilon;
      }

      x -= (alpha * numeratorX) / denominatorX;
      y -= (alpha * numeratorY) / denominatorY;

      coordinates.push({ x, y });
    }
    return coordinates;
  }

  //function for contour plot
  let f = (x, y) => x ** 2 + y ** 2;

  // for example (x, y) => {return { x: 2 * x, y: 2 * y };};
  let grad = (x, y) => {
    return { x: 2 * x, y: 2 * y };
  };

  let epochs = 250;

  //gradient descent
  let vanillaCoordinates = calculatePath(epochs, f, grad, -20, 20, 0.01);
  let momentumCoordinates = calculatePath(epochs, f, grad, 20, 20, 0.01, true);

  /*--------------------------------------------*/
  //squished contours and rmsprop
  let epochs2 = 100;
  let f2 = (x, y) => x ** 2 + 9 * y ** 2;
  let grad2 = (x, y) => {
    return { x: 2 * x, y: 18 * y };
  };
  let vanillaCoordinates2 = calculatePath(epochs2, f2, grad2, -1, -1, 0.01);
  let momentumCoordinates2 = calculatePath(
    epochs2,
    f2,
    grad2,
    1,
    1,
    0.01,
    true,
    false
  );
  let rmsCoordinates2 = calculatePath(
    epochs2,
    f2,
    grad2,
    -1,
    1,
    0.01,
    false,
    true
  );
</script>

<svelte:head>
  <title>Optimizers - World4AI</title>
  <meta
    name="description"
    content="Optimizers like momentum gradient descent, RMSProp and adam have several advantages over vanilla gradient descent and can speed up training significantly. Switching out optimizers in PyTorch is relatively easy and usually requires switching just a single line of code."
  />
</svelte:head>

<h1>Optimizers</h1>
<div class="separator" />

<Container>
  <p>
    In deep learning the specific gradient descent algorithm is called an
    <Highlight>optimizer</Highlight>. So far we have only really looked at the
    plain vanilla gradient descent optimizer called <code>SGD</code>, short for
    stochastic gradient descent.
  </p>
  <PythonCode code={`optimizer = optim.SGD(model.parameters(), lr=0.01)`} />
  <p>
    With each batch we use the backpropagation algorithm to calculate the
    gradient vector <Latex>{String.raw`\mathbf{\nabla}_w`}</Latex>. The gradient
    descent optimizer directly subtracts the gradient, scaled by the learning
    rate <Latex>\alpha</Latex>, from the weight vector <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > without any further adjustments.
  </p>
  <Latex
    >{String.raw`\mathbf{w}_{t+1} := \mathbf{w}_t - \alpha \mathbf{\nabla}_w`}</Latex
  >
  <p>
    As you can probably guess this is not the only and by far not the fastest
    approach available. Other optimizers have been developed over time that
    generally converge a lot faster.
  </p>
  <div class="separator" />

  <h2>Momentum</h2>
  <p>
    The plain vanilla gradient descent algorithm lacks any form of memory. This
    optimizer only takes the gradient direction from the current batch into
    consideration and disregards any past gradient calculations.
  </p>
  <p>
    When we use stochastic gradient descent with momentum on the other hand, we
    keep a moving average of the past directions and use that average
    additionally to the current gradient to adjust the weights.
  </p>
  <Latex
    >{String.raw`\mathbf{m_t} = \beta \mathbf{m}_{t-1} + (1 - \beta) \mathbf{\nabla}_w `}</Latex
  >
  <p>
    At each timestep <Latex>{String.raw`t`}</Latex> we calculate the momentum vector
    <Latex>{String.raw`\mathbf{m}_t`}</Latex> as a weighted average of the previous
    momentum <Latex>{String.raw`\mathbf{m}_{t-1}`}</Latex> and the current gradient
    <Latex>{String.raw`\mathbf{\nabla}_w`}</Latex>, where
    <Latex>\beta</Latex> is usually around 0.9. As the initial momentum vector <Latex
      >{String.raw`\mathbf{m}_0`}</Latex
    > is essentially empty, deep learning frameworks like PyTorch initialize the
    vector by setting the momentum to the actual gradient vector.
  </p>
  <Latex>{String.raw`\mathbf{m}_0 = \mathbf{\nabla}_w`}</Latex>
  <p>
    When we apply gradient descent, we do not use the gradient vector <Latex
      >\nabla</Latex
    > directly to adjust the weights of the neural network, but use momentum instead.
  </p>
  <Latex
    >{String.raw`\mathbf{w}_{t+1} := \mathbf{w}_t - \alpha \mathbf{m}_t`}</Latex
  >
  <p>
    In PyTorch we can use gradient descent with momentum by passing an
    additional argument to the <code>SGD</code> object.
  </p>
  <PythonCode
    code={`optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`}
  />
  <p>
    But why is momentum actually useful? Below we see the same example with the
    local minimum, that we studied the first time we encountered gradient
    descent. The example showed, that gradient descent will get stuck in a local
    minimum. Gradient descent with momentum on the other has a chance to escape
    the local minimum.
  </p>

  <ButtonContainer>
    <StepButton on:click={localGradientDescent} />
  </ButtonContainer>
  <Plot
    width="450"
    height="450"
    maxWidth="500"
    domain={[-3, 7]}
    range={[-40, 120]}
    padding={{ top: 20, right: 20, bottom: 20, left: 25 }}
  >
    <Ticks
      xTicks={[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]}
      yTicks={[-40, -20, 0, 20, 40, 60, 80, 100, 120]}
    />
    <Path data={localMinimumData} />
    <Circle data={localPoint} />
  </Plot>

  <p>
    Even when we are dealing with a direct path towards the minimum without any
    saddle points and local minima, the momentum optimizer will build
    acceleration and converge faster towards the minimum. Below we compare the
    convergence speed between simple stochastic gradient descent and momentum
    for <Latex>{String.raw`x^2 + y^2`}</Latex>. The momentum based approach
    arrives faster at the optimum.
  </p>
  <Plot
    width="600"
    height="600"
    maxWidth="700"
    domain={[-1, 1]}
    range={[-1, 1]}
    padding={{ top: 0, right: 0, bottom: 0, left: 0 }}
  >
    <Contour {f} thresholds={d3.range(-2, 2, 0.1).map((i) => i)} />
    <Path data={vanillaCoordinates} stroke="2" strokeDashArray={"4 4"} />
    <Path
      data={momentumCoordinates}
      color="var(--main-color-4)"
      strokeDashArray={"4 4"}
      stroke="2"
    />
    <XLabel text="x" type="latex" y="580" fontSize="15" />
    <YLabel text="y" type="latex" y="300" fontSize="15" />
    <Legend
      coordinates={{ x: -0.9, y: -0.85 }}
      legendColor="black"
      text="Vanilla Gradient Descent"
    />
    <Legend
      coordinates={{ x: -0.9, y: -0.9 }}
      text="Gradient Descent With Momentum"
      legendColor="var(--main-color-4)"
    />
  </Plot>
  <div class="separator" />

  <h2>RMSProp</h2>
  <p>
    Adaptive optimizers, like RMSProp<InternalLink type="note" id="1" />, do not
    adjust speed per se, but determine a better direction for gradient descent.
    If we are dealing with a bowl shaped loss function for example, the
    gradients will not be symmetrical. That means that we will approach the
    optimal value not in a direct line, but rather in a zig zagging manner.
  </p>
  <Plot
    width="1000"
    height="200"
    maxWidth="700"
    domain={[-1, 1]}
    range={[-1, 1]}
    padding={{ top: 5, right: 5, bottom: 20, left: 30 }}
  >
    <Ellipse
      data={[{ x: 0, y: 0 }]}
      radiusX="480"
      radiusY="80"
      color="var(--main-color-4)"
    />
    <Ellipse data={[{ x: 0, y: 0 }]} radiusX="240" radiusY="40" color="none" />
    <Ellipse data={[{ x: 0, y: 0 }]} radiusX="120" radiusY="20" color="none" />
    <Ellipse data={[{ x: 0, y: 0 }]} radiusX="60" radiusY="10" color="none" />
    <Ellipse data={[{ x: 0, y: 0 }]} radiusX="30" radiusY="5" color="none" />
    <Path
      data={[
        { x: -0.85, y: -0.55 },
        { x: -0.8, y: +0.65 },
        { x: -0.75, y: -0.6 },
        { x: -0.7, y: +0.62 },
        { x: -0.65, y: -0.6 },
        { x: -0.6, y: +0.57 },
        { x: -0.5, y: -0.5 },
        { x: -0.4, y: +0.5 },
      ]}
      stroke="2"
      strokeDashArray="2 4"
    />
    <Ticks xTicks={[-1, 0, 1]} yTicks={[-1, 0, 1]} />
  </Plot>
  <p>
    We would like to move more in a the x direction and less in the y direction,
    which would result in a straight line towards the optimium. Theoretically we
    could offset the zig zag by using an individual learning rate for each of
    the weights, but given that there are million of weights in modern deep
    learning, this approach is not feasable. Adaptive optimizers scale each
    gradient in such a way, that we approach the optimum in a much straighter
    line. These optimizers allow to use a single learning rate for the whole
    neural network.
  </p>
  <p>
    Similar to momentum, RMSProp (root mean squared prop) calcualtes a moving
    average, but instead of tracking the gradient, we track the squared
    gradient.
  </p>
  <Latex
    >{String.raw`\mathbf{d_t} = \beta_2 \mathbf{d}_{t-1} + (1 - \beta_2) \mathbf{\nabla}_w^2`}</Latex
  >
  <p>
    This root of the vector <Latex>{String.raw`\mathbf{d}`}</Latex> is used to scale
    the gradient. This causes the gradients to get similar in magnitute (which creates
    a straighter line), while still following the general direction that is encoded
    in the moving average.
  </p>
  <Latex
    >{String.raw`\mathbf{w}_{t+1} := \mathbf{w}_t - \alpha \dfrac{\mathbf{\nabla}_w}{\sqrt{\mathbf{d}_t} + \epsilon}`}</Latex
  >
  <p>
    The <Latex>\epsilon</Latex> varialble is a very small positive number that is
    used in order to avoid divisions by 0.
  </p>
  <p>
    Below we compare vanilla gradient descent, gradient descent with momentum
    and RMSProp on a loss function with an elongated form. While the simple
    gradient descent and momentum gradient descent approach the optimum in a
    curved manner, RMSProp takes basically a straight route. Also notice, that
    momentum can overshoot due to gained speed and needs some time to reverse
    direction.
  </p>
  <Plot
    width="600"
    height="600"
    maxWidth="700"
    domain={[-1, 1]}
    range={[-1, 1]}
    padding={{ top: 5, right: 5, bottom: 20, left: 30 }}
  >
    <Contour f={f2} thresholds={d3.range(-16, 16.1, 0.5).map((i) => i)} />
    <Ticks xTicks={d3.range(-2, 2)} yTicks={d3.range(-2, 2)} />
    <Path data={vanillaCoordinates2} stroke="4" strokeDashArray={"3 6"} />
    <Path
      data={momentumCoordinates2}
      color="var(--main-color-2)"
      stroke="4"
      strokeDashArray={"3 6"}
    />
    <Path
      data={rmsCoordinates2}
      color="var(--main-color-3)"
      stroke="4"
      strokeDashArray={"3 6"}
    />
    <Circle data={[{ x: 0, y: 0 }]} />
    <Legend
      coordinates={{ x: -0.9, y: -0.85 }}
      legendColor="black"
      text="Vanilla Gradient Descent"
    />
    <Legend
      coordinates={{ x: -0.9, y: -0.9 }}
      text="Gradient Descent With Momentum"
      legendColor="var(--main-color-2)"
    />
    <Legend
      coordinates={{ x: -0.9, y: -0.95 }}
      text="RMSProp"
      legendColor="var(--main-color-3)"
    />
  </Plot>
  <p>
    The api for all optimizers in PyTorch is identical, so we can simply replace
    the <code>SGD</code> object with the <code>RMSprop</code> object and we are good
    to go.
  </p>
  <PythonCode code={`optimizer = optim.RMSprop(model.parameters(), lr=0.01)`} />
  <div class="separator" />

  <h2>Adam</h2>
  <p>
    Adam<InternalLink id="1" type="reference" /> is the combination of momentum and
    adaptive learning. If you look at the equations below, you will not find any
    new concepts. We calculate moving averages of the gradients and the squared gradients.
    The RMSProp style scaling is not applied directly to the gradient vector, instead
    we scale the momentum vector and use the result to adjust the weights.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
    \mathbf{m_t} &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{\nabla}_w \\
    \mathbf{d_t} &= \beta_2 \mathbf{d}_{t-1} + (1 - \beta_2) \mathbf{\nabla}_w^2 \\
    \mathbf{w}_{t+1} & := \mathbf{w}_t - \alpha \dfrac{\mathbf{m}_t}{\sqrt{\mathbf{d}_t} + \epsilon}
  \end{aligned}
    `}</Latex
  >
  <p>
    Adam (and its derivatives) is probably the most used optimizer at this point
    in time. If you don't have any specific reason to use a different optimizer,
    use adam.
  </p>
  <p>We can implement the adam optimizer in PyTorch the following way.</p>
  <PythonCode code={`optimizer = optim.Adam(model.parameters(), lr=0.01)`} />
</Container>
<Footer {references} {notes} />
