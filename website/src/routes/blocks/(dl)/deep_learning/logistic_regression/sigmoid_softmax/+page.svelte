<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Plot from "$lib/Plot.svelte";
  import Slider from "$lib/Slider.svelte";
  import Highlight from "$lib/Highlight.svelte";

  const regressionData = [
    [
      { x: 0, y: 0 },
      { x: -2, y: 0 },
      { x: -1, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 },
    ],
    [
      { x: 6, y: 1 },
      { x: 7, y: 1 },
      { x: 8, y: 1 },
      { x: 9, y: 1 },
      { x: 10, y: 1 },
      { x: 11, y: 1 },
      { x: 12, y: 1 },
    ],
  ];

  const limitedRegressionData = [
    [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 },
    ],
    [
      { x: 6, y: 1 },
      { x: 7, y: 1 },
      { x: 8, y: 1 },
      { x: 9, y: 1 },
      { x: 10, y: 1 },
    ],
  ];

  // data for threshhold function
  let threshholdData = [];
  for (let i = -2; i <= 12; i++) {
    let x = i;
    let y = 1;
    if (x <= 5) {
      y = 0;
    }
    threshholdData.push({ x, y });

    // needed to make the threshhold absolute vertical
    if (x == 5) {
      threshholdData.push({ x: 5, y: 1 });
    }
  }

  // data for sigmoid function
  let logisticData = [];
  for (let i = -6; i <= 6; i += 0.1) {
    let x = i;
    let y = 1 / (1 + Math.exp(-x));
    logisticData.push({ x, y });
  }

  //function to demonstrate how we can change the sigmoid function based on inputs
  let weight = 1;
  let bias = 0;
  let movingSigmoidData = [];

  function recalculateSigmoid() {
    movingSigmoidData = [];
    for (let i = -16; i <= 16; i += 0.1) {
      let x = i;
      let z = weight * x + bias;

      let y = 1 / (1 + Math.exp(-z));
      movingSigmoidData.push({ x, y });
    }
  }

  recalculateSigmoid();
  $: weight && recalculateSigmoid();
  $: bias && recalculateSigmoid();

  // this section deals with the demonstration of the decision boundary
  let decisionData = [
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

  let decisionW1 = -0.15;
  let decisionW2 = 0.2;
  let decisionB = -0.01;

  let decisionPathsData = [];
  function updateDecisionPathsData() {
    let x1 = 0;
    let x2 = 1;

    let y1 = (-decisionB - decisionW1 * x1) / decisionW2;
    let y2 = (-decisionB - decisionW1 * x2) / decisionW2;
    decisionPathsData = [
      { x: x1, y: y1 },
      { x: x2, y: y2 },
    ];
  }

  updateDecisionPathsData();
  $: decisionW1 && updateDecisionPathsData();
  $: decisionW2 && updateDecisionPathsData();
  $: decisionB && updateDecisionPathsData();
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Sigmoid and Softmax</title>
  <meta
    name="description"
    content="The sigmoid activation function bounds the outputs between 0 and 1, allowing the results to be interpreted as probabilities."
  />
</svelte:head>

<Container>
  <h1>Sigmoid and Softmax</h1>
  <div class="separator" />
  <p>
    Let us start from the basic assumption, that we want to come up with a
    classification algorithm. Similar to linear regression there should be
    learnable parameters <Latex>{String.raw`\mathbf{w}`}</Latex> and <Latex
      >b</Latex
    >, but unlike with linear regression the output of the algorithm should be
    the probability to belong to a particular category.
  </p>
  <div class="separator" />

  <h2>Linear Regression</h2>
  <p>
    We might start by implementing simple linear regression without any
    adjustments. Let us assume that at the moment we face two possible
    categories and therefore the output <Latex>y</Latex> is either 0 or 1. We therefore
    need to train a model that produces values between 0 and 1. These values can
    be regarded as probabilities to belong to the category 1. If the output is 0.3
    for example, the model predicts that based on the features we are dealing with
    category 1 with 30% probability and with 70% probability we are dealing with
    category 0.
  </p>
  <p>
    We could draw a line just like the one below and at first glance this seems
    to be a reasonable approach. Higher values of some feature correspond to a
    higher probability to belong to the "blue" category and lower values of the
    same feature correspond to a lower probability.
  </p>
  <Plot
    pointsData={limitedRegressionData}
    pathsData={[
      { x: 0, y: 0 },
      { x: 10, y: 1 },
    ]}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: 0,
      maxX: 10,
      minY: 0,
      maxY: 1,
      xLabel: "Feature",
      yLabel: "Label",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 5,
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
      numTicks: 6,
    }}
  />
  <p>
    We could also get into trouble with linear regression and our regression
    line could produce results that are above 1 or below 0, values that can not
    be interpreted as probabilities. Especially when our model faces data that
    contains new unforseen features, the linear regression model would break
    apart.
  </p>
  <Plot
    pointsData={regressionData}
    pathsData={[
      { x: -2, y: -0.2 },
      { x: 12, y: 1.2 },
    ]}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -2,
      maxX: 12,
      minY: -0.2,
      maxY: 1.2,
      xLabel: "Feature",
      yLabel: "Label",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 5,
      xTicks: [-1, 1, 3, 5, 7, 9, 11],
      yTicks: [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
    }}
  />
  <div class="separator" />

  <h2>Threshold Activation</h2>
  <p>
    In our second attempt to construct a classification algorithm we could the
    original threshold activation function that was used in the McCulloch and
    Pitts neuron.
  </p>
  <p>
    We could use the threshold of 5, which would mean that each sample with a
    feature value above 5 is classified into the "blue" category and the rest
    would be classified as the "red" category.
  </p>
  <Latex
    >{String.raw`
      f(x) = 
      \left\{ 
      \begin{array}{rcl}
      0 & for & x \leq 5 \\ 
      1 & for & x > 5 \\
      \end{array}
      \right.
    `}</Latex
  >
  <Plot
    pathsData={threshholdData}
    pointsData={regressionData}
    config={{
      minX: -2,
      maxX: 12,
      minY: 0,
      maxY: 1,
      xLabel: "Feature",
      yLabel: "Label",
      xTicks: [-1, 1, 3, 5, 7, 9, 11],
      yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1],
    }}
  />
  <p>
    While this rule perfectly separates the data into the two categories, the
    threshold function is not differentiable. A non differentiable function
    would prevent us from applying gradient descent, which would limit our
    ability to learn optimal weights and biases.
  </p>
  <div class="separator" />

  <h2>Sigmoid</h2>
  <p>
    The sigmoid function <Latex
      >{String.raw`\sigma(x) = \dfrac{1}{1 + e^{-x}}`}</Latex
    > is an S shaped function that is commonly used in machine learning to produce
    probabilites, because the function does not display any of the problems that
    we faced with the two approaches above.
  </p>

  <Plot
    pathsData={logisticData}
    config={{
      minX: -6,
      maxX: 6,
      minY: 0,
      maxY: 1,
      xLabel: "x",
      yLabel: "f(x)",
      xTicks: [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
      yTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }}
  />
  <p>
    <Latex>\sigma(x)</Latex> is always bounded between 0 and 1, no matter how large
    or how negative the inputs are. This allows us to interpret the results as probabilities.
  </p>
  <p>
    The sigmoid is a softer version of the threshold function. It smoothly
    changes between the probabilities. The function is therefore differentiable,
    which allows us to use gradient descent to learn the weights and biases.
  </p>
  <p>
    Usually the output of 0.5 (50%) is regarded as the cutoff point. That would
    mean that inputs above 0 would be classified as category one and inputs
    below 0 would be classified as category 0.
  </p>
  <p>
    In practice we combine linear regression with a sigmoid function, which
    forms the basis for logistic regression. The output of logistic regression
    is used as the input into the sigmoid.
  </p>
  <Latex
    >{String.raw`\hat{y} = \sigma(\mathbf{w}, x) = \dfrac{1}{1 + e^{-z}}`}</Latex
  >, where <Latex>{String.raw`z = \mathbf{x} \mathbf{w}^T + b`}</Latex>
  <p>
    This procedure allows to learn parameters <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and <Latex>b</Latex> which align the true categories <Latex
      >{String.raw`\mathbf{y}`}</Latex
    >
    with the probabilities <Latex>{String.raw`\mathbf{\hat{y}}`}</Latex>. Below
    is an interactive example that lets you change the weight and the bias and
    observe how the probabilities would change based on the inputs. Using both
    sliders you can move and rotate the probabilities as much as you want. Try
    to find parameters that would fit the data.
  </p>
  <Plot
    pathsData={movingSigmoidData}
    pointsData={regressionData}
    config={{
      minX: -16,
      maxX: 16,
      minY: 0,
      maxY: 1,
      xLabel: "x",
      yLabel: "f(x)",
      xTicks: [
        -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16,
      ],
      yTicks: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }}
  />
  <div class="flex-container">
    <div><Latex>w</Latex>: {weight}</div>
    <Slider bind:value={weight} min={-5} max={5} step={0.1} />
  </div>
  <div class="flex-container">
    <div><Latex>b</Latex>: {bias}</div>
    <Slider bind:value={bias} min={-30} max={30} step={0.1} />
  </div>

  <p>
    When we are dealing with a classification problem, we are trying to draw a
    decision boundary between the different classes in order to separate the
    data as good as possible. In the below example we have a classification
    problem with two features and two classes. We utilize logistic regression
    (the sigmoid function) with two weights <Latex>w_1</Latex>, <Latex
      >w_2</Latex
    > and the bias <Latex>b</Latex> to draw a boundary. The boundary represents the
    exact cutoff, the 50% probability. On the one side of the boundary you would
    have
    <Latex>{String.raw`\dfrac{1}{1 + e^{-(x_1w_1 + x_2w_2 + b)}} > 0.5`}</Latex
    >, while on the other side of the boundary you have <Latex
      >{String.raw`\dfrac{1}{1 + e^{-(x_1w_1 + x_2w_2 + b)}} < 0.5`}</Latex
    >. By changing the weights and the bias you can rotate and move the decision
    boundary respectively.
  </p>
  <Plot
    pointsData={decisionData}
    pathsData={decisionPathsData}
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
  <div class="flex-container">
    <div><Latex>w_1</Latex>: {decisionW1}</div>
    <Slider bind:value={decisionW1} min={-5} max={5} step={0.01} />
  </div>
  <div class="flex-container">
    <div><Latex>w_2</Latex>: {decisionW2}</div>
    <Slider bind:value={decisionW2} min={-5} max={5} step={0.01} />
  </div>
  <div class="flex-container">
    <div><Latex>b</Latex>: {decisionB}</div>
    <Slider bind:value={decisionB} min={-5} max={5} step={0.01} />
  </div>
  <p>
    When we apply gradient descent to logistic regression, essentially we are
    adjusting the weights and the bias to separate the data.
  </p>
  <div class="separator" />
  <h2>Softmax</h2>
  <p>
    Before we move on to the next chapter, let us shortly discuss what function
    can be used if we are faced with more than two categories. The explanations
    below will not make full sense until we reach the chapter that covers the
    artificial neural networks, but this is a good place to make a short
    introduction.
  </p>
  <p>
    Let us assume, that we face a classification problem with d possible
    categories. Our goal is to calculate the probabilities to belong to each of
    these categories. The softmax function takes a <Latex>d</Latex> dimensional vector
    <Latex>{String.raw`\mathbf{z}`}</Latex> and returns a vector of the same size
    that contains the corresponding probabilities.
  </p>
  <Latex
    >{String.raw`
softmax(\mathbf{z}) = 
softmax
\begin{pmatrix}
\begin{bmatrix}
   z_1  \\
   z_2 \\ 
   z_3 \\ 
   \vdots \\
   z_d 
\end{bmatrix}
\end{pmatrix}
=
\begin{bmatrix}
   p_1 \\
   p_2 \\ 
   p_3 \\ 
   \vdots \\
   p_d
\end{bmatrix}
\\
    `}</Latex
  >
  <p>
    If we had four categories for example, the results might look as follows.
  </p>
  <Latex
    >{String.raw`
softmax(\mathbf{z}) = 

\begin{bmatrix}
   0.05 \\
   0.1 \\ 
   0.8 \\ 
   0.05
\end{bmatrix}
\\
    `}</Latex
  >
  <p>
    Given these numbers, we would assume that it is most likely that the
    features belong to the category Nr. 3.
  </p>
  <p>
    The values <Latex>{String.raw`\mathbf{z}`}</Latex> that are used as input into
    the softmax function are called <Highlight>logits</Highlight>. You can
    imagine that each of the logits is a separate linear regression of the form <Latex
      >{String.raw`z = \mathbf{x} \mathbf{w}^T + b`}</Latex
    > and that we have <Latex>d</Latex> linear regressions, as many as there are
    categories. The weights and the biases for each of the logits are independent
    of each other, which means that each of the linear regressions produces different
    results, which leads to a different probability.
  </p>
  <p>
    We calculate the probability for the <Latex>k</Latex> of <Latex>d</Latex> categories
    using the following softmax equation.
  </p>
  <Latex
    >{String.raw`\large softmax(z_k) = \dfrac{e^{z_k}}{\sum_d e^{z_d}}`}</Latex
  >
  <p>
    Similar to sigmoid function, the softmax function has several advantageous
    properties. The equation for example guarantees, that the sum of
    probabilities is exactly 1, thus avoiding any violations of the law of
    probabilities. Additionally as the name suggest the function is "soft",
    which indicates that it is differentiable and can be used in gradient
    descent.
  </p>
  <div class="separator" />
</Container>

<style>
  .flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .flex-container div {
    width: 100px;
  }
</style>
