<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Plot from "$lib/Plot.svelte";
  import Button from "$lib/Button.svelte";

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
</script>

<h1>Gradient Descent</h1>
<div class="separator" />

<Container>
  <p>
    Our goal is to find weights <Latex>{String.raw`\mathbf{w}`}</Latex> and the bias
    <Latex>b</Latex> that minimize the cross-entropy loss in a binary classification
    problem.
  </p>

  <Latex
    >{String.raw`
    \arg\min_{\mathbf{w}, b}H(p, q) = - \sum_i \Big[y^{(i)} \log \sigma(\mathbf{w},b)) + (1 - y^{(i)}) (1 - \log(\sigma(\mathbf{w}, b)) \Big] \\
  `}</Latex
  >
  <p>
    Similar to linear regression, logistic regression relies on gradient
    descent. While the derivative is more complex, the chain rule still allows
    us to relatively easily find the optimum.
  </p>
  <p>
    Ultimately we are interested in the gradient of the loss function <Latex
      >H(p, w)</Latex
    > with respect to the weight vector <Latex
      >{String.raw`\nabla_{\mathbf{x}}`}</Latex
    > and the derivative with respect to the bias <Latex
      >{String.raw`\dfrac{\partial}{\partial b} H(p,x)`}</Latex
    >.
  </p>
  <p>
    Once we have those we can use the same gradient descent procedure that we
    used in linear regression.
  </p>
  <Latex
    >{String.raw`\large \mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w \\`}</Latex
  >
  <Latex
    >{String.raw`\large b_{t+1} \coloneqq b_t - \alpha \dfrac{\partial}{\partial b} \\`}</Latex
  >
  <p>
    That meas that all we have to do is to figure out how we can calculate the
    gradients, the rest of the procedure does not differ from linear regression.
  </p>
  <Latex
    >{String.raw`
    H(p, q) = - \sum_i \Big[y^{(i)} \log a + (1 - y^{(i)}) \log(1 - a)\Big] \\
  `}</Latex
  >
  <div class="separator" />
  <Latex
    >{String.raw`
    \dfrac{\partial}{\partial a}H(p, q) = -\Big(y^{(i)} \dfrac{1}{a} - (1 - y^{(i)}) \dfrac{1}{1 - a}\Big)
  `}</Latex
  >
  <div class="separator" />
  <Latex>{String.raw`a = \sigma(z) = \dfrac{1}{1 + e^{-z}}`}</Latex>
  <div class="separator" />
  <Latex
    >{String.raw`\dfrac{\partial}{\partial z}a = \sigma(z) (1 - \sigma(z)) `}</Latex
  >
  <div class="separator" />
  <Latex>{String.raw`z = \mathbf{w^Tx}+b`}</Latex>
  <div class="separator" />
  <Latex>{String.raw`\dfrac{\partial}{\partial w_j} z = x_j`}</Latex>
  <div class="separator" />
  <Latex>{String.raw`\dfrac{\partial}{\partial b} z = 1`}</Latex>
  <div class="separator" />

  <Plot
    {pointsData}
    {pathsData}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: 0,
      maxX: 1,
      minY: 0,
      maxY: 1,
      xLabel: "Feature 1",
      yLabel: "Feature 2",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 5,
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
    }}
  />
  <div class="parameters yellow">
    <div class="flex">
      <div class="left">
        <p><strong>Cross Entropy</strong> <Latex>H(p,q)</Latex>:</p>
        <p><strong>Weight</strong> <Latex>w_1</Latex>:</p>
        <p><strong>Weight</strong> <Latex>w_2</Latex>:</p>
        <p><strong>Bias</strong> <Latex>b</Latex>:</p>
      </div>
      <div class="right">
        <p><strong>{crossEntropy.toFixed(2)}</strong></p>
        <p><strong>{w1.toFixed(2)}</strong></p>
        <p><strong>{w2.toFixed(2)}</strong></p>
        <p><strong>{b.toFixed(2)}</strong></p>
      </div>
    </div>
  </div>
  <Button value={"Gradient Descent Step"} on:click={gradientDescentStep} />
</Container>

<style>
  .parameters {
    width: 50%;
    padding: 5px 10px;
    margin-bottom: 5px;
  }

  div p {
    margin: 0;
    border-bottom: 1px solid black;
  }

  .flex {
    display: flex;
    flex-direction: row;
  }

  .left {
    flex-grow: 1;
    margin-right: 20px;
  }

  .right {
    flex-basis: 40px;
  }
</style>
