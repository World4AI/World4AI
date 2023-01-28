<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Slider from "$lib/Slider.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";

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
  <title>Sigmoid and Softmax - World4AI</title>
  <meta
    name="description"
    content="The sigmoid activation function bounds the outputs between 0 and 1, allowing the results to be interpreted as probabilities. The softmax activation function works in a similar manner, but unlike the sigmoid works for more than 2 categories."
  />
</svelte:head>

<Container>
  <h1>Sigmoid and Softmax</h1>
  <div class="separator" />
  <p>
    Let us assume that no classification algorithms have been invented yet and
    that we want to come up with the first classification algorithm. We are
    assuming that there should be learnable parameters <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and <Latex>b</Latex> and the output of the model should correspond the probability
    to belong to one of two categories. We will expand our ideas to more categories
    at a later step.
  </p>
  <div class="separator" />

  <h2>Linear Regression</h2>
  <p>
    Let's see what happens when we simply use linear regression for
    classification tasks.
  </p>
  <p>
    Our dataset contains two classes. We assign each of the classes eather the
    label 0 or 1 and we need to train a model that produces values between 0 and
    1. These values can be regarded as probabilities to belong to the category
    1. If the output is 0.3 for example, the model predicts that we are dealing
    with category 1 with 30% probability and with 70% probability we are dealing
    with category 0.
  </p>
  <p>
    We could draw a line just like the one below and at first glance this seems
    to be a reasonable approach. Higher values of some feature correspond to a
    higher probability to belong to the "blue" category and lower values of the
    same feature correspond to a lower probability.
  </p>
  <Plot width={500} height={250} maxWidth={800} domain={[0, 10]} range={[0, 1]}>
    <Ticks
      xTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
      yTicks={[0, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} />
    <YLabel text="Label" fontSize={15} />
    <Path
      data={[
        { x: 0, y: 0 },
        { x: 10, y: 1 },
      ]}
    />
    <Circle data={limitedRegressionData[0]} />
    <Circle data={limitedRegressionData[1]} color={"var(--main-color-2)"} />
  </Plot>
  <p>
    While linear regression might work during training, when we start facing new
    datapoints we might get into trouble, because our model can theoretically
    produce results that are above 1 or below 0, values that can not be
    interpreted as probabilities.
  </p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[-2, 12]}
    range={[0, 1]}
  >
    <Ticks
      xTicks={[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
      yTicks={[0, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} />
    <YLabel text="Label" fontSize={15} />
    <Path
      data={[
        { x: -2, y: -0.2 },
        { x: 12, y: 1.2 },
      ]}
    />
    <Circle data={regressionData[0]} />
    <Circle data={regressionData[1]} color={"var(--main-color-2)"} />
  </Plot>
  <Alert type="warning">
    Never use linear regression for classification tasks. There is no built-in
    mechanism that prevents linear regression from producing nonsencical
    probabiilty results.
  </Alert>
  <div class="separator" />

  <h2>Threshold Activation</h2>
  <p>
    In our second attempt to construct a classification algorithm we could the
    use original threshold activation function that was used in the McCulloch
    and Pitts neuron.
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
    width={500}
    height={250}
    maxWidth={800}
    domain={[-2, 12]}
    range={[0, 1]}
  >
    <Ticks
      xTicks={[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
      yTicks={[0, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} />
    <YLabel text="Label" fontSize={15} />
    <Path data={threshholdData} />
    <Circle data={regressionData[0]} />
    <Circle data={regressionData[1]} color={"var(--main-color-2)"} />
  </Plot>
  <p>
    While this rule perfectly separates the data into the two categories, the
    threshold function is not differentiable. A non differentiable function
    would prevent us from applying gradient descent, which would limit our
    ability to learn optimal weights and biases.
  </p>
  <div class="separator" />

  <h2>Sigmoid</h2>
  <p>
    The <Highlight>sigmoid</Highlight> function <Latex
      >{String.raw`\sigma(x) = \dfrac{1}{1 + e^{-x}}`}</Latex
    > is an S shaped function that is commonly used in machine learning to produce
    probabilites.
  </p>

  <Plot width={500} height={250} maxWidth={800} domain={[-6, 6]} range={[0, 1]}>
    <Ticks
      xTicks={[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="x" fontSize={10} type="latex" />
    <YLabel text="f(x)" fontSize={10} x={-2} type="latex" />
    <Path data={logisticData} />
  </Plot>
  <p>
    The sigmoid does not display problems that we faced with the two approaches
    above.
    <Latex>\sigma(x)</Latex> is always bounded between 0 and 1, no matter how large
    or how negative the inputs are. This allows us to interpret the results as probabilities.
    The sigmoid is also a softer version of the threshold function. It smoothly changes
    between the probabilities. The function is therefore differentiable, which allows
    us to use gradient descent to learn the weights and biases.
  </p>
  <p>
    Usually the output of 0.5 (50%) is regarded as the cutoff point. That would
    mean that inputs above 0.5 would be classified as category one and inputs
    below 0.5 would be classified as category 0.
  </p>
  <p>
    In practice we combine linear regression with a sigmoid function, which
    forms the basis for logistic regression. The output of logistic regression
    is used as the input into the sigmoid.
  </p>
  <Alert type="info">
    Logistic regression uses linear regression
    <Latex>{String.raw`z = \mathbf{x} \mathbf{w}^T + b`}</Latex> as an input into
    the sigmoid
    <Latex>{String.raw`\hat{y} = \dfrac{1}{1 + e^{-z}}`}</Latex>
  </Alert>
  <p>
    This procedure allows us to learn parameters <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and <Latex>b</Latex> which align the true categories <Latex
      >{String.raw`\mathbf{y}`}</Latex
    >
    with predicted probabilities <Latex>{String.raw`\mathbf{\hat{y}}`}</Latex>.
    Below is an interactive example that lets you change the weight and the
    bias. Observe how probabilities change based on the inputs. Using both
    sliders you can move and rotate the probabilities as much as you want. Try
    to find parameters that would fit the data.
  </p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[-16, 16]}
    range={[0, 1]}
  >
    <Ticks
      xTicks={[
        -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16,
      ]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="x" fontSize={10} type="latex" />
    <YLabel text="f(x)" fontSize={10} x={-2} type="latex" />
    <Path data={movingSigmoidData} />
    <Circle data={regressionData[0]} />
    <Circle data={regressionData[1]} color="var(--main-color-2)" />
  </Plot>
  <Slider label="Weight" labelId="weight" showValue={true} bind:value={weight} min={-5} max={5} step={0.1} />
  <Slider label="Bias" labelId="bias" showValue={true} bind:value={bias} min={-30} max={30} step={0.1} />

    <p>
      When we are dealing with a classification problem, we are trying to draw a
      decision boundary between the different classes in order to separate the
      data as good as possible. In the below example we have a classification
      problem with two features and two classes. We utilize logistic regression
      (the sigmoid function) with two weights <Latex>w_1</Latex>, <Latex
        >w_2</Latex
      > and the bias <Latex>b</Latex> to draw a boundary. The boundary represents
      the exact cutoff, the 50% probability. On the one side of the boundary you
      would have
      <Latex
        >{String.raw`\dfrac{1}{1 + e^{-(x_1w_1 + x_2w_2 + b)}} > 0.5`}</Latex
      >, while on the other side of the boundary you have <Latex
        >{String.raw`\dfrac{1}{1 + e^{-(x_1w_1 + x_2w_2 + b)}} < 0.5`}</Latex
      >. By changing the weights and the bias you can rotate and move the
      decision boundary respectively.
    </p>
    <Plot
      width={500}
      height={250}
      maxWidth={800}
      domain={[0, 1]}
      range={[0, 1]}
    >
      <Ticks
        xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        xOffset={-15}
        yOffset={15}
      />
      <XLabel text="Feature 1" fontSize={15} />
      <YLabel text="Feature 2" fontSize={15} />
      <Path data={decisionPathsData} />
      <Circle data={decisionData[0]} />
      <Circle data={decisionData[1]} color="var(--main-color-2)" />
    </Plot>

    <Slider label={"Weight 1"} labelId={'w1'} showValue={true} bind:value={decisionW1} min={-5} max={5} step={0.01} />
    <Slider label={"Weight 2"} labelId={'w2'} showValue={true} bind:value={decisionW2} min={-5} max={5} step={0.01} />
    <Slider label={"Bias"} labelId={'b'} showValue={true} bind:value={decisionB} min={-5} max={5} step={0.01} />
    <p>
      When we apply gradient descent to logistic regression, essentially we are
      adjusting the weights and the bias to shift the decision boundary.
    </p>
    <div class="separator" />

    <h2>Softmax</h2>
    <p>
      Before we move on to the next section, let us shortly discuss what
      function can be used if we are faced with more than two categories.
    </p>
    <p>
      Let us assume, that we face a classification problem with <Latex>d</Latex>
      possible categories. Our goal is to calculate the probabilities to belong to
      each of these categories. The <Highlight>softmax</Highlight> function takes
      a <Latex>d</Latex> dimensional vector
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
      imagine that each of the <Latex>d</Latex> logits is a separate linear regression
      of the form <Latex>{String.raw`z = \mathbf{x} \mathbf{w}^T + b`}</Latex>.
    </p>
    <p>
      We calculate the probability for the <Latex>k</Latex> of <Latex>d</Latex> categories
      using the following softmax equation.
    </p>
    <Latex>{String.raw`softmax(z_k) = \dfrac{e^{z_k}}{\sum_d e^{z_d}}`}</Latex>
    <p>
      Similar to the sigmoid function, the softmax function has several
      advantageous properties. The equation for example guarantees, that the sum
      of probabilities is exactly 1, thus avoiding any violations of the law of
      probabilities. Additionally as the name suggest the function is "soft",
      which indicates that it is differentiable and can be used in gradient
      descent.
    </p>
    <div class="separator" />
  </Container
>
