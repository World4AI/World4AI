<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Plot from "$lib/Plot.svelte";
  import Slider from "$lib/Slider.svelte";

  let linearData = [[]];
  let b = 0;
  let w = 5;
  for (let i = -100; i < 100; i++) {
    let x = i;
    let y = b + w * (x + Math.random() * 20 - 10);
    linearData[0].push({ x, y });
  }

  let nonlinearData = [[]];
  for (let i = -100; i < 100; i++) {
    let x = i + 1;
    let y = (x + Math.random() * 10 - 10) ** 2;
    nonlinearData[0].push({ x, y });
  }

  let lines = [
    { x: -100, y: 0 },
    { x: 100, y: 0 },
  ];

  let estimatedBias = -200;
  let estimatedWeight = -100;

  function calculatePoints(estimatedBias, estimatedWeight) {
    let y1 = estimatedBias + estimatedWeight * lines[0].x;
    let y2 = estimatedBias + estimatedWeight * lines[1].x;
    lines[0].y = y1;
    lines[1].y = y2;
    lines = lines;
  }

  $: calculatePoints(estimatedBias, estimatedWeight);
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Linear Model</title>
  <meta
    name="description"
    content="A linear model allows us to model the data using a line (or a hyperplane) in the coordinate system."
  />
</svelte:head>

<Container>
  <h1>Linear Model</h1>
  <div class="separator" />
  <p>
    The term "linear regression" consists of two words, that fully describe the
    type of model we are dealing with: <Highlight>linear</Highlight> and <Highlight
      >regression</Highlight
    >. The "regression" part signifies that our model predicts a continuous
    label based on some given features (we are not dealing with classification).
    The "linear" part suggests that linear regression can only model the
    relationship between features and label in a linear fashion. To understand
    what a "linear" relationship means we present two examples below.
  </p>

  <p>
    In the first scatterplot we could plot a line that goes from the coordinates
    of (-100, -500) and goes to coordinates of (100, 500). While there is some
    randomness in the data, the line would depict the relationship between the
    feature and the label relatively well. When we get new data points we can
    use the line to predict the label and be relatively confident regarding the
    outcome.
  </p>
  <Plot
    pointsData={linearData}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -100,
      maxX: 100,
      minY: -500,
      maxY: 500,
      xLabel: "Feature",
      yLabel: "Label",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 3,
      colors: ["var(--main-color-1)", "var(--main-color-2)"],
      numTicks: 11,
    }}
  />
  <p>
    In contrast the data in the following scatterplot represents a nonlinear
    relationship between the feature and the label. Theoretically there is
    nothing that stops us from using linear regression for the below problem,
    but there are better alternatives (like neural networks) for non linear
    problems.
  </p>
  <Plot
    pointsData={nonlinearData}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -100,
      maxX: 100,
      minY: 0,
      maxY: 10000,
      xLabel: "Feature",
      yLabel: "Label",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 3,
      colors: ["var(--main-color-1)", "var(--main-color-2)"],
      numTicks: 11,
    }}
  />
  <p>
    From basic math we know, that in the two dimensional space we can draw a
    line using <Latex>y = wx + b</Latex>. For our purposes <Latex>x</Latex> is the
    single feature, <Latex>y</Latex> is the label, <Latex>w</Latex> is the weight
    that we use to scale the feature and <Latex>b</Latex> is the bias.
  </p>
  <p>
    While we can easily understand that the feature <Latex>x</Latex> is the input
    of our equation and the label <Latex>y</Latex> is the output of the equation,
    we have a harder time imagining what role the weight <Latex>w</Latex> and the
    bias <Latex>b</Latex> play in the equation. Simply put the weight determines
    the rotation (slope) of the line while the bias determines the position.
  </p>
  <p>
    Below we present an interactive example to demonstrate the impact of the
    weight and the bias on the line. You can move the two slides to change the
    weight and the bias by moving the sliders. Observe what we mean when we say
    rotation and position.
  </p>
  <Plot
    pointsData={linearData}
    pathsData={lines}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -100,
      maxX: 100,
      minY: -500,
      maxY: 500,
      xLabel: "Feature",
      yLabel: "Label",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 3,
      colors: ["var(--main-color-1)", "var(--main-color-2)"],
      numTicks: 11,
    }}
  />
  <p>The weight <Latex>w</Latex> is: {estimatedWeight}</p>
  <Slider bind:value={estimatedWeight} min={-200} max={200} />
  <p>The bias <Latex>b</Latex> is: {estimatedBias}</p>
  <Slider bind:value={estimatedBias} min={-500} max={500} />
  <p>
    We used the weight <Latex>w</Latex> of 5 and a bias <Latex>b</Latex> of 0 plus
    some randomness to generate the data above. When you played with sliders you
    must have come relatively close. The main takeaway from this is that the weight
    and the bias are learnable parameters. The linear regression algorithm provides
    us with a way to find those parameters.
  </p>
  <p>
    In practice we rarely deal with a dataset where we only have one feuature.
    In that case our equation looks like follows.
  </p>
  <Latex>\large y = w_1x_1 + w_2x_2 + ... + w_nx_n + b</Latex>
  <p>
    We can also use a more compact form and write the equation as a dot product.
  </p>
  <Latex
    >{String.raw`\large y = \mathbf{w} \cdot \mathbf{x} + b \text{, where}`}</Latex
  >
  <Latex
    >{String.raw`
    \mathbf{x} = 
\begin{bmatrix}
   x_1  \\
   x_2  \\
   \vdots \\
   x_n
\end{bmatrix}
\mathbf{w} = 
\begin{bmatrix}
   w_1  \\
   w_2  \\
   \vdots \\
   w_n
\end{bmatrix}
`}
  </Latex>
  <p>
    In a three dimensional space we calculate a two dimensional plane that
    divides the coordinate system into two regions. This procedure is harder to
    imagine for more than 3 dimensions, but we still create a plane (a so called
    hyperplane) in the space. The weights are used to rotate the hyperplane
    while the bias moves the plane.
  </p>
  <p>
    Usually we draw a hat over the <Latex>y</Latex> value to indicate that we are
    dealning with a prediction from a model. Therefore our equation looks as below.
  </p>
  <Latex>{String.raw`\large \hat{y} = \mathbf{w} \cdot \mathbf{x} + b`}</Latex>
  <p>
    The next lectures are going to cover how the learning procedure works in
    linear regression. The main takeaway from this chapter should be the visual
    intuition of weights and biases and the notational foundation that we
    covered.
  </p>
  <div class="separator" />
</Container>
