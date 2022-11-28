<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte"; 
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import XLabel from "$lib/plt/XLabel.svelte"; 
  import YLabel from "$lib/plt/YLabel.svelte"; 
  import Path from "$lib/plt/Path.svelte"; 
  import Circle from "$lib/plt/Circle.svelte"; 
  import Legend from "$lib/plt/Legend.svelte"; 
  import Text from "$lib/plt/Text.svelte"; 

  //Difference in Gradients Demonstration
  let graphDataLoss = [];
  let mseData0 = [];
  let mseData1 = [];
  let ceData0 = [];
  let ceData1 = [];

  let graphGradLoss = [];
  let mseGrad0 = [];
  let mseGrad1 = [];
  let ceGrad0 = [];
  let ceGrad1 = [];

  for (let i = 0; i <= 1; i += 0.01) {
    let dataPoint;
    let gradPoint;
    let grad;
    //mse
    // target is 0
    let mse;
    mse = (0 - i) ** 2;
    grad = -2 * (0 - i);
    dataPoint = { x: i, y: mse };
    gradPoint = { x: i, y: grad };
    mseData0.push(dataPoint);
    mseGrad0.push(gradPoint);

    //target is 1
    mse = (1 - i) ** 2;
    grad = -2 * (1 - i);
    dataPoint = { x: i, y: mse };
    mseData1.push(dataPoint);
    gradPoint = { x: i, y: grad };
    mseGrad1.push(gradPoint);

    //cross-entropy
    let ce;
    if (i !== 0 && i !== 1) {
      // target is 0
      ce = -Math.log(1 - i);
      grad = 1 / (1 - i);
      dataPoint = { x: i, y: ce };
      gradPoint = { x: i, y: grad };
      ceData0.push(dataPoint);
      ceGrad0.push(gradPoint);
      // target = 1
      ce = -Math.log(i);
      grad = -1 / i;
      dataPoint = { x: i, y: ce };
      gradPoint = { x: i, y: grad };
      ceData1.push(dataPoint);
      ceGrad1.push(gradPoint);
    }
  }
  graphDataLoss.push(mseData0);
  graphDataLoss.push(mseData1);
  graphDataLoss.push(ceData0);
  graphDataLoss.push(ceData1);

  graphGradLoss.push(mseGrad0);
  graphGradLoss.push(mseGrad1);
  graphGradLoss.push(ceGrad0);
  graphGradLoss.push(ceGrad1);

  // Gradient Descent Demonstrations
  let w1 = -0.15;
  let w2 = 0.2;
  let b = -0.01;
  let alpha = 0.5;

  let dw1 = 0;
  let dw2 = 0;
  let db = 0;
  let crossEntropy = 0;
  let numPoints = 30;

  let pointsData = [
    [
      { x: 0, y: 0 },
      { x: 0.1, y: 0.23 },
      { x: 0.25, y: 0.93 },
      { x: 0.15, y: 0.63 },
      { x: 0.25, y: 0.13 },
      { x: 0.1, y: 0.93 },
      { x: 0.12, y: 0.53 },
      { x: 0.32, y: 0.23 },
      { x: 0.22, y: 0.5 },
      { x: 0.49, y: 0.1 },
      { x: 0.45, y: 0.3 },
      { x: 0.4, y: 0.7 },
      { x: 0.35, y: 0.5 },
      { x: 0.25, y: 0.7 },
      { x: 0.2, y: 0.2 },
    ],
    [
      { x: 1, y: 1 },
      { x: 0.75, y: 0.89 },
      { x: 0.75, y: 0.75 },
      { x: 0.95, y: 0.7 },
      { x: 0.85, y: 0.7 },
      { x: 0.65, y: 0.8 },
      { x: 0.85, y: 0.4 },
      { x: 0.75, y: 0.25 },
      { x: 0.75, y: 0.55 },
      { x: 0.95, y: 0.35 },
      { x: 0.85, y: 0.15 },
      { x: 0.85, y: 0.95 },
      { x: 0.9, y: 0.55 },
      { x: 0.9, y: 0.28 },
      { x: 0.98, y: 0.95 },
    ],
  ];

  let pathsData = [];
  function updatePathsData() {
    let x1 = 0;
    let x2 = 1;

    let y1 = (-b - w1 * x1) / w2;
    let y2 = (-b - w1 * x2) / w2;
    pathsData = [
      { x: x1, y: y1 },
      { x: x2, y: y2 },
    ];
  }

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function calculateGradients() {
    dw1 = 0;
    dw2 = 0;
    db = 0;
    crossEntropy = 0;

    pointsData.forEach((category, idx) => {
      category.forEach((point) => {
        let dzdw1 = point.x;
        let dzdw2 = point.y;
        let z = point.x * w1 + point.y * w2 + b;
        let a = sigmoid(z);
        crossEntropy += -(idx * Math.log(a) + (1 - idx) * Math.log(1 - a));

        let dadz = a * (1 - a);
        let dHda = -(idx * (1 / a) - (1 - idx) * (1 / (1 - a)));

        dw1 += dHda * dadz * dzdw1;
        dw2 += dHda * dadz * dzdw2;
        db += dHda * dadz;
      });
    });
    dw1 /= numPoints;
    dw2 /= numPoints;
    db /= numPoints;
  }

  function gradientDescentStep() {
    //take gradient descent step
    w1 -= alpha * dw1;
    w2 -= alpha * dw2;
    b -= alpha * db;

    updatePathsData();
    calculateGradients();
  }

  calculateGradients();
  updatePathsData();

  function train() {
    gradientDescentStep();
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Logistic Regression Gradient Descent</title>
  <meta
    name="description"
    content="The optimal weights and biases for logistic regression can be obtained by the means of gradient descent. The gradients are calcualted using the chain rule."
  />
</svelte:head>

<h1>Gradient Descent</h1>
<div class="separator" />

<Container>
  <h2>Cross-Entropy vs Mean Squared Error</h2>
  <p>
    The cross-entropy is almost exclusively used as the loss function for
    classification tasks, but it is not obvious why we can not use the mean
    squared error. Actually we can, but as we will see shortly, the
    cross-entropy is a more convenient measure of loss for classification tasks.
  </p>
  <p>
    For this discusson we will deal with a single sample and distinquish between
    two cases: the label <Latex>y</Latex> is either 0 or 1. If the label equals to
    0, the mean squared error is <Latex>{String.raw`(0 - \sigma)^2`}</Latex> while
    the cross-entropy is<Latex>{String.raw`-\log(1 - \sigma)`}</Latex>. Both
    losses increase as the predicted probability <Latex>\sigma</Latex> grows. If
    the true label <Latex>1</Latex> is 1 on the other hand the mean squared error
    is <Latex>{String.raw`(1 - \sigma)^2`}</Latex> and the cross-entropy is<Latex
      >{String.raw`-\log(\sigma)`}</Latex
    >. In both cases the error decreases when the predicted probability <Latex
      >{String.raw`\sigma`}</Latex
    > grows.
  </p>
  <p>
    Below we plot the mean squared error and the cross-entropy based on the
    predicted probability <Latex>\sigma</Latex>. The red plot depicts the mean
    squared error, while the blue plot depicts the cross-entropy. There are two
    plots for each of the losses, one for each value of the target.
  </p>

  <Plot width={500} height={250} maxWidth={800} domain={[0, 1]} range={[0, 3]}>
    <Ticks xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
           yTicks={[0, 1, 2, 3]} 
           xOffset={-15} 
           yOffset={15}/>
    <XLabel text="Predicted Probability" fontSize={15} />
    <YLabel text="Error" fontSize={15} />
    <Path data={graphDataLoss[0]} color="var(--main-color-1)" />
    <Path data={graphDataLoss[1]} color="var(--main-color-1)"/>
    <Path data={graphDataLoss[2]} color="var(--main-color-2)"/>
    <Path data={graphDataLoss[3]} color="var(--main-color-2)"/>
    <Legend coordinates={{x: 0.3, y: 2.8}} legendColor="var(--main-color-2)" text="Cross Entropy"/>
    <Legend coordinates={{x: 0.3, y: 2.5}} legendColor="var(--main-color-1)" text="Mean Squared Error"/>
  </Plot>
  <p>
    The mean squared error and the cross-entropy start at the same position, but
    the difference in errors starts to grow as the predicted probability starts
    to deviate from the true label. The cross-entropy punishes
    misclassifications with a much higher loss, than the mean squared error.
    When we deal with probabilities the difference between the label and the
    predicted probability can not be larger than 1. That means that the mean
    squared error also can not grow beyond 1. The logarithm on the other hand
    literally explodes when the value starts approaching 0.
  </p>
  <p>
    This behaviour can also be observed when we draw the predicted probability
    against the derivative of the loss function. While the derivatives of the
    mean squared error are linear, the cross-entropy derivatives grow
    exponentially when the quality of predictions deteriorates.
  </p>
  <Plot width={500} height={250} maxWidth={800} domain={[0, 1]} range={[-10, 10]}>
    <Ticks xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
           yTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]} 
           xOffset={-15} 
           yOffset={15}/>
    <XLabel text="Predicted Probability" fontSize={15} />
    <YLabel text="Derivative" fontSize={15} />
    <Path data={graphGradLoss[0]} color="var(--main-color-1)" />
    <Path data={graphGradLoss[1]} color="var(--main-color-1)"/>
    <Path data={graphGradLoss[2]} color="var(--main-color-2)"/>
    <Path data={graphGradLoss[3]} color="var(--main-color-2)"/>
    <Legend coordinates={{x: 0.3, y: 8}} legendColor="var(--main-color-2)" text="Cross Entropy"/>
    <Legend coordinates={{x: 0.3, y: 6}} legendColor="var(--main-color-1)" text="Mean Squared Error"/>
  </Plot>
  <p>
    The exponential growth of derivatives implies, that the gradient descent
    algorithm will take much larger steps, when the classification predictions
    are way off, thereby converging at a higher rate.
  </p>

  <div class="separator" />
  <h2>The Algorithm</h2>
  <p>
    Our goal is to find weights <Latex>{String.raw`\mathbf{w}`}</Latex> and the bias
    <Latex>b</Latex> that minimize the cross-entropy loss in a binary classification
    problem.
  </p>

  <Latex
    >{String.raw`
      \underset{\mathbf{w}, b} {\arg\min} L \\ 
      L =  - \dfrac{1}{n} \sum_i \Big[y^{(i)} \log \sigma + (1 - y^{(i)}) \log(1 - \sigma) \Big] \\
  `}</Latex
  >
  <p>
    Similar to linear regression, logistic regression relies on gradient
    descent. While the derivative is slightly more complex, the chain rule still
    allows us to relatively easily find the optimum. Ultimately we are
    interested in the gradient of the loss function <Latex>L</Latex> with respect
    to the weight vector <Latex>{String.raw`\nabla_{\mathbf{w}}`}</Latex> and the
    derivative with respect to the bias <Latex
      >{String.raw`\dfrac{\partial}{\partial b} L`}</Latex
    >. Once we have those we can use the same gradient descent procedure that we
    used in linear regression.
  </p>
  <Latex
    >{String.raw`\mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w `}</Latex
  >
  <br />
  <Latex
    >{String.raw`b_{t+1} \coloneqq b_t - \alpha \dfrac{\partial}{\partial b} `}</Latex
  >
  <p>
    We can calculate gradients for individual samples and calculate the mean
    afterwards. That reduces the loss of an individual sample to the following
    equation.
  </p>
  <Latex
    >{String.raw`
    L = - \Big[y \log \sigma + (1 - y) \log(1 - \sigma)\Big] \\
  `}</Latex
  >
  <p>
    We start by calculating the derivative of the loss with respect to the
    sigmoid output.
  </p>
  <Latex
    >{String.raw`
    \dfrac{\partial}{\partial \sigma}L = - \Big[y^{(i)} \dfrac{1}{\sigma}  - (1 - y^{(i)}) \dfrac{1}{(1 - \sigma)}\Big] \\
  `}</Latex
  >
  <p>
    Next we calculate the derivative of the sigmoid <Latex
      >{String.raw`\sigma(z) = \dfrac{1}{1 + e^{-z}}`}</Latex
    > with respect to the net input <Latex>z</Latex>.
  </p>

  <Latex
    >{String.raw`\dfrac{\partial}{\partial z}\sigma = \sigma(z) (1 - \sigma(z)) `}</Latex
  >
  <p>
    The derivative of the sigmoid function is relatively straightforward, but
    the derivation process is somewhat mathematically involved. It is not
    necessary to know the exact steps how we arrive at the derivative, but if
    you are interested, below we provide the necessary steps.
  </p>
  <Latex
    >{String.raw`
    \begin{aligned}
    \dfrac{\partial}{\partial z}\sigma &= \dfrac{\partial}{\partial z} \dfrac{1}{1 + e^{-z}} \\
    &= \dfrac{\partial}{\partial z} ({1 + e^{-z}})^{-1} \\
    &= -({1 + e^{-z}})^{-2} * -e^{-z} \\
    &= \dfrac{ e^{-z}}{({1 + e^{-z}})^{-2}} \\
    &= \dfrac{1}{{1 + e^{-z}}} \dfrac{ e^{-z}}{{1 + e^{-z}}} \\
    &= \dfrac{1}{{1 + e^{-z}}} \dfrac{ 1 + e^{-z} - 1}{{1 + e^{-z}}} \\
    &= \dfrac{1}{{1 + e^{-z}}} \Big(\dfrac{ 1 + e^{-z}}{1 + e^{-z}} - \dfrac{1}{{1 + e^{-z}}}\Big) \\
    &= \dfrac{1}{{1 + e^{-z}}} \Big(1 - \dfrac{1}{{1 + e^{-z}}}\Big) \\
    &= \sigma(z) (1 - \sigma(z)) 
    \end{aligned}
    `}</Latex
  >
  <p>
    The rest of derivations are do not differ from those that we used in linear
    regression. We calculate the gradients of the net input <Latex
      >{String.raw`z = \mathbf{xw^T}+b`}</Latex
    > with respect to the individual weights and the bias applying the chain rule.
  </p>
  <Latex>{String.raw`\dfrac{\partial}{\partial w_j} z = x_j`}</Latex>
  <br />
  <Latex>{String.raw`\dfrac{\partial}{\partial b} z = 1`}</Latex>
  <p>
    Finally we can utilize the chain rule to calculate the derivatives of the
    loss with respect to the weights and the bias. When we are dealing with
    batch or minibatch gradient descent we collect the gradients for several
    samples <Latex>n</Latex> and calculate the mean, which is utilized in gradient
    descent.
  </p>
  <Latex
    >{String.raw`\dfrac{\partial L}{\partial w_j} = \dfrac{1}{n} \sum_i^n \dfrac{\partial L}{\partial \sigma^{(i)}} \dfrac{\partial \sigma^{(i)}}{\partial z^{(i)}} \dfrac{\partial z^{(i)}}{\partial w_j^{(i)}}`}</Latex
  >
  <p>
    In the interactive example below we demonstrate the gradient descent
    algorithm for logistic regression. This is the same example that you tried
    to solve manually in a previous chapter. Start the algorithm and observe how
    the loss decreases over time.
  </p>
  <ButtonContainer>
    <PlayButton f={train} delta={100} />
  </ButtonContainer>

  <Plot width={500} height={250} maxWidth={800} domain={[0, 1]} range={[0, 1.2]}>
    <Ticks xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
           yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
           xOffset={-15} 
           yOffset={15}/>
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Path data={pathsData} />
    <Circle data={pointsData[0]}/>
    <Circle data={pointsData[1]} color={"var(--main-color-2)"}/>
    <Text text="L: {crossEntropy.toFixed(2)}" x={0} y={1.4} />
    <Text text="w1: {w1.toFixed(2)}" x={0} y={1.3} />
    <Text text="w2: {w1.toFixed(2)}" x={0} y={1.2} />
    <Text text="b: {b.toFixed(2)}" x={0} y={1.1} />
  </Plot>
  <p>
    The gradient descent algorithm learns to separate the data in a matter of
    seconds.
  </p>
  <div class="separator" />
</Container>
