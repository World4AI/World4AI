<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import PlayButton from "$lib/PlayButton.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Contour from "$lib/plt/Contour.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";

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

  //function for contour plot
  let f = (x, y) => x ** 2 + y ** 2;

  // for example (x, y) => {return { x: 2 * x, y: 2 * y };};
  let gradient = (x, y) => {
    return { x: 2 * x, y: 2 * y };
  };

  let epochs = 250;

  //gradient descent
  let vanillaCoordinates = [];
  let momentumCoordinates = [];

  let vanillaStartingX = -100;
  let vanillaStartingY = 100;
  let momentumStartingX = 100;
  let momentumStartingY = 100;

  function calculateGradients() {
    let learningRate = 0.01;
    let beta = 0.91;
    let tau = 0.1;
    let momentum = { x: 0, y: 0 };

    vanillaCoordinates = [];
    momentumCoordinates = [];
    let xVanilla;
    let yVanilla;
    let xMomentum;
    let yMomentum;
    for (let i = 0; i < epochs; i++) {
      if (i === 0) {
        xVanilla = vanillaStartingX;
        yVanilla = vanillaStartingY;
        vanillaCoordinates.push({ x: xVanilla, y: yVanilla });

        xMomentum = momentumStartingX;
        yMomentum = momentumStartingY;
        momentumCoordinates.push({ x: xMomentum, y: yMomentum });
      } else {
        xVanilla = vanillaCoordinates[i].x;
        yVanilla = vanillaCoordinates[i].y;

        xMomentum = momentumCoordinates[i].x;
        yMomentum = momentumCoordinates[i].y;
      }

      let vanillaGrads = gradient(xVanilla, yVanilla);

      xVanilla -= learningRate * vanillaGrads.x;
      yVanilla -= learningRate * vanillaGrads.y;
      let vanillaCoordinate = { x: xVanilla, y: yVanilla };
      vanillaCoordinates.push(vanillaCoordinate);

      let momentumGrads = gradient(xMomentum, yMomentum);

      if (i === 0) {
        momentum.x = momentumGrads.x;
        momentum.y = momentumGrads.y;
      }

      momentum.x = beta * momentum.x + tau * momentumGrads.x;
      momentum.y = beta * momentum.y + tau * momentumGrads.y;

      xMomentum -= learningRate * momentum.x;
      yMomentum -= learningRate * momentum.y;

      let momentumCoordinate = { x: xMomentum, y: yMomentum };
      momentumCoordinates.push(momentumCoordinate);
    }
  }
  calculateGradients();
</script>

<h1>Optimizers</h1>
<div class="separator" />

<Container>
  <p>
    In deep learning the specific gradient descent algorithm is called an
    <Highlight>optimizer</Highlight>. So far we have only really looked at the
    plain vanilla gradient descent optimizer. At each batch we use the
    backpropagation algorithm to calculate the gradient vector <Latex
      >{String.raw`\mathbf{\nabla}_w`}</Latex
    >. The gradient descent optimizer directly subtracts the gradient, scaled by
    the learning rate <Latex>\alpha</Latex>, from the weight vector <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > without any further adjustments.
  </p>
  <Latex
    >{String.raw`\mathbf{w}_{t+1} := \mathbf{w}_t - \alpha \mathbf{\nabla}_w`}</Latex
  >
  <p>
    As you can imagine this is not the only and by far not the fastest approach
    available. Other optimizers have been developed over time that generally
    converge a lot faster. Also oftentimes the plain vanilla gradient descent
    algorithm will get stuck in a saddle point, while modern approaches will
    find a way to overcome the saddle point.
  </p>
  <div class="separator" />

  <h2>Momentum</h2>
  <p>
    The plain vanilla gradient descent algorithm lacks any form of memory. If
    the derivative for a variable is +1 at timestep 1 and -1 at timestep 2, the
    optimizer will disregard the past direction and only move into the -1
    direction of the variable.
  </p>
  <p>
    Momentum on the other hand keeps a moving average of the past directions and
    uses those additionally to the current gradient when applying gradient
    descent.
  </p>
  <Latex
    >{String.raw`\mathbf{m_t} = \beta \mathbf{m}_{t-1} + (1 - \beta) \mathbf{\nabla}_w `}</Latex
  >
  <p>
    The vector <Latex>{String.raw`\mathbf{m}_t`}</Latex> contains the momentum vector,
    which is build as a weighted average of the past momentum <Latex
      >{String.raw`\mathbf{m}_{t-1}`}</Latex
    > and the current gradient <Latex>{String.raw`\mathbf{\nabla}_w`}</Latex> . At
    each timepoint <Latex>t</Latex> we multiply the momentum vector from the previous
    period <Latex>{String.raw`\mathbf{m}_{t-1}`}</Latex> with the momentum factor
    <Latex>\beta</Latex>. Usually this factor is around 0.9. We scale the
    current gradient vector <Latex>{String.raw`\mathbf{\nabla}_w`}</Latex> by <Latex
      >1-\beta</Latex
    >. The sum is used in the calculation of gradient descent.
  </p>
  <Latex
    >{String.raw`\mathbf{w}_{t+1} := \mathbf{w}_t - \alpha \mathbf{m}_t`}</Latex
  >
  <p>
    As the momentum vector <Latex>{String.raw`\mathbf{m}_0`}</Latex> is essentially
    empty, the deep learning frameworks like PyTorch and Keras initialize the vector
    by setting the momentum to the actual gradient vector.
  </p>
  <Latex>{String.raw`\mathbf{m}_0 = \mathbf{\nabla}_w`}</Latex>
  <p>
    Below we see the same example with the local minimum, that we studied the
    first time we encountered gradient descent. The example showed, that
    gradient descent will get stuck in a local minimum. Gradient descent with
    momentum on the other hand has a chance, has a chance to escape the local
    minimum.
  </p>

  <PlayButton on:click={localGradientDescent} />
  <Plot
    width="450"
    height="450"
    maxWidth="500"
    domain={[-3, 7]}
    range={[-40, 120]}
    padding={{ top: 40, right: 40, bottom: 40, left: 50 }}
  >
    <Ticks
      xTicks={[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]}
      yTicks={[-40, -20, 0, 20, 40, 60, 80, 100, 120]}
    />
    <XLabel text="x" type="latex" />
    <YLabel text="f(x)" type="latex" />
    <Path data={localMinimumData} />
    <Circle data={localPoint} />
  </Plot>

  <Plot
    width="700"
    height="700"
    maxWidth="700"
    domain={[-1, 1]}
    range={[-1, 1]}
    padding={{ top: 0, right: 0, bottom: 0, left: 0 }}
  >
    <Contour {f} />
    <Ticks xTicks={[-1, -0.5, 0, 0.5, 1]} yTicks={[-1, -0.5, 0, 0.5, 1]} />
    <Path data={vanillaCoordinates} stroke="2" strokeDashArray={"4 4"} />
    <Path
      data={momentumCoordinates}
      color="var(--main-color-4)"
      strokeDashArray={"4 4"}
      stroke="2"
    />
    <!--
    {#each Array(epochs + 1) as _, idx}
      <Circle data={[vanillaCoordinates[idx]]} />
      <Circle data={[momentumCoordinates[idx]]} />
    {/each}
    -->
  </Plot>
  <div class="separator" />

  <h2>RMSProp</h2>
  <div class="separator" />

  <h2>Adam</h2>
  <div class="separator" />
</Container>
