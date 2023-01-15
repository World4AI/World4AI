<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // table library
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";

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
  <title>Logistic Regression Gradient Descent - World4AI</title>
  <meta
    name="description"
    content="The optimal weights and biases for logistic regression can be obtained by the means of gradient descent. The gradients are calcualted using the chain rule."
  />
</svelte:head>

<h1>Gradient Descent</h1>
<div class="separator" />

<Container>
  <p>
    We finally have the means to find weights <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and the bias
    <Latex>b</Latex> that minimize the binary cross-entropy loss.
  </p>
  <p>
    Let's remind ourselves, that calculation the cross-entropy loss requires
    several steps. In the first step we calculate the so called <Highlight
      >net input</Highlight
    ><Latex>z</Latex>, where
    <Latex>{String.raw`z = \mathbf{xw^T}+b`}</Latex>. The net input is used as
    an input into the sigmoid function <Latex>\sigma</Latex>, where <Latex
      >{String.raw`\sigma(z) = \dfrac{1}{1 + e^{-z}}`}</Latex
    >. The output of the sigmoid is the probability that the features belong to
    the class with the label <Latex>1</Latex>. Finally we use the output of the
    sigmoid as an input into the cross-entropy loss <Latex>L</Latex>, where
    <Latex
      >{String.raw`
      L =  - \dfrac{1}{n} \sum_i \Big[y^{(i)} \log \sigma(z^{(i)}) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \Big] \\
  `}</Latex
    >
  </p>
  <p>
    Similar to linear regression, logistic regression relies on gradient
    descent. While the derivative is slightly more complex, the chain rule still
    allows us to relatively easily find the optimum. Given the three composed
    functions describe above, we can use the chain rule the following way.
  </p>
  <Latex
    >{String.raw`
    \dfrac{\partial L}{\partial w_j} = \dfrac{1}{n} \sum_i^n \dfrac{\partial L}{\partial \sigma^{(i)}} \dfrac{\partial \sigma^{(i)}}{\partial z^{(i)}} \dfrac{\partial z^{(i)}}{\partial w_j^{(i)}} \\
    \dfrac{\partial L}{\partial b} = \dfrac{1}{n} \sum_i^n \dfrac{\partial L}{\partial \sigma^{(i)}} \dfrac{\partial \sigma^{(i)}}{\partial z^{(i)}} \dfrac{\partial z^{(i)}}{\partial b} \\
    `}</Latex
  >
  <p>Once we have those gradients we can use the gradient descent procedure.</p>
  <Latex
    >{String.raw`\mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w `}</Latex
  >
  <br />
  <Latex
    >{String.raw`b_{t+1} \coloneqq b_t - \alpha \dfrac{\partial}{\partial b} `}</Latex
  >
  <p>
    Let's once again assume, that we are dealing with a single sample to
    simplify notation. As the derivative of the sum is the sum of derivatives,
    you can easily extend our explanations to a whole dataset.
  </p>
  <Latex
    >{String.raw`
    L = - \Big[y \log(z) \sigma + (1 - y) \log(1 - \sigma(z))\Big] \\
  `}</Latex
  >
  <p>
    We start by calculating the derivative of the loss with respect to the
    sigmoid output <Latex>\sigma</Latex>. Using basic rules of calculus we get
    the following result.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    \dfrac{\partial}{\partial \sigma}L = - \Big[y \dfrac{1}{\sigma(z)}  - (1 - y) \dfrac{1}{(1 - \sigma(z))}\Big] \\
  `}</Latex
    >
  </div>
  <p>
    Next we calculate the derivative of the sigmoid <Latex
      >{String.raw`\sigma(z) = \dfrac{1}{1 + e^{-z}}`}</Latex
    > with respect to the net input <Latex>z</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\dfrac{\partial}{\partial z}\sigma = \sigma(z) (1 - \sigma(z)) `}</Latex
    >
  </div>
  <p>
    The derivative of the sigmoid function is relatively straightforward, but
    the derivation process is somewhat mathematically involved. It is not
    necessary to know the exact steps how we arrive at the derivative, but if
    you are interested, below we provide the necessary steps.
  </p>
  <Alert type="info">
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
  </Alert>
  <p>
    The rest of derivations do not differ from those that we used in linear
    regression. We calculate the gradients of the net input <Latex
      >{String.raw`z = \mathbf{xw^T}+b`}</Latex
    > with respect to the individual weights and the bias.
  </p>
  <p class="flex justify-center items-center">
    <Latex
      >{String.raw`\dfrac{\partial}{\partial w_j} z = x_j \\
    \dfrac{\partial}{\partial b} z = 1`}</Latex
    >
  </p>
  <p>
    If we wanted use NumPy and Python to implement logistic regression the
    result might look as our example below.
  </p>
  <PythonCode
    code={`for epoch in range(epochs):
    # 1. calculate predicted probabilities
    z = X @ w.T + b
    sigma = 1 / (1 + np.exp(-z))
    
    # 2. calculate the partial derivatives 
    dH_dsigma = -(y * (1 / sigma) - (1 - y) * (1 / (1 - sigma)))
    dsigma_dz = sigma * (1 - sigma)
    dz_dx = X
    dz_db = 1
    
    # 3. apply the chain rule
    grad_w = (dH_dsigma * dsigma_dz * dz_dx).mean(axis=0)
    grad_b = (dH_dsigma * dsigma_dz * dz_db).mean()
    
    # 4. apply batch gradient descent
    w = w - alpha * grad_w
    b = b - alpha * grad_b`}
  />
  `

  <p>
    In the interactive example below we demonstrate the gradient descent
    algorithm for logistic regression. This is the same example that you tried
    to solve manually in a previous chapter. Start the algorithm and observe how
    the loss decreases over time.
  </p>
  <ButtonContainer>
    <PlayButton f={train} delta={100} />
  </ButtonContainer>
  <Table>
    <TableHead>
      <Row>
        <HeaderEntry>Variable</HeaderEntry>
        <HeaderEntry>Value</HeaderEntry>
      </Row>
    </TableHead>
    <TableBody>
      <Row>
        <DataEntry>
          <Latex>L</Latex>
        </DataEntry>
        <DataEntry>
          {crossEntropy.toFixed(2)}
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          <Latex>w_1</Latex>
        </DataEntry>
        <DataEntry>
          {w1.toFixed(2)}
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          <Latex>w_2</Latex>
        </DataEntry>
        <DataEntry>
          {w2.toFixed(2)}
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          <Latex>b</Latex>
        </DataEntry>
        <DataEntry>
          {b.toFixed(2)}
        </DataEntry>
      </Row>
    </TableBody>
  </Table>
  <Plot width={500} height={250} maxWidth={800} domain={[0, 1]} range={[0, 1]}>
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Path data={pathsData} />
    <Circle data={pointsData[0]} />
    <Circle data={pointsData[1]} color={"var(--main-color-2)"} />
  </Plot>
  <p>
    The gradient descent algorithm learns to separate the data in a matter of
    seconds.
  </p>
  <div class="separator" />
</Container>
