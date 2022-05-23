<script>
  import Container from "$lib/Container.svelte";
  import Plot from "$lib/Plot.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import NeuralNetwork from "./_nonlinear/NeuralNetwork.svelte";

  let numbers = 50;
  let notes = [
    `The decision boundary in the visualisations is caclulated approximately by dividing the graph into ${numbers} rows and ${numbers} columns and drawing a box with the corresponding class. If the decision boundaries don't look smooth, this is due to the approximative nature of the calculation.`,
  ];

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

  let heatmapData = [];
  for (let i = 0; i < numbers; i++) {
    for (let j = 0; j < numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let classification;
      if (x + y > 1) {
        classification = 0;
      } else {
        classification = 1;
      }
      let coordinate = { x, y, class: classification };
      heatmapData.push(coordinate);
    }
  }

  let heatmapData2 = [];
  for (let i = 0; i < numbers; i++) {
    for (let j = 0; j < numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let classification;
      if ((x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.12) {
        classification = 0;
      } else {
        classification = 1;
      }
      let coordinate = { x, y, class: classification };
      heatmapData2.push(coordinate);
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
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Nonlinear Problems</title>
  <meta
    name="description"
    content="Most interesting problems in machine learning are highly nonlinear. Neural networks are capable of solving such problems, if the network uses nonlinear activation functions and at least 1 hidden layer."
  />
</svelte:head>

<h1>Nonlinear Problems</h1>
<div class="separator" />

<Container>
  <p>
    We have arrived at a point in our studies, where we can start to understand
    neural networks, but there are several questions we should ask ourselves and
    try to answer before we move on to the technicalities of neural networks.
  </p>
  <div class="separator" />
  <div class="info">
    1. Why do we require neural network when we can do regression using linear
    regression and classification using logistic regression?
  </div>
  <p>
    In the example below we have a classification problem with two classes and
    two features. By visually inspecting the dataset, the human brain can
    quickly separate the data the by imagining a circle between the two classes.
  </p>
  <Plot {pointsData} {config} />
  <p>
    If you look at the logistic regression lecture, you will notice that
    logistic regression produces a linear decision boundary<InternalLink
      type="note"
      id="1"
    />. While this might be sufficient for some problems, in our case we would
    misclassify approximately half of the data. Logistic regression is only able
    to solve linear problems, while the data below clearly depicts a nonlinear
    problem.
  </p>
  <Plot {pointsData} {heatmapData} {config} />
  <p>
    A neural network on the other hand can theoretically generate an adequate
    decision boundary for nonlinear problems, which would result in a much
    better accuracy.
  </p>
  <Plot {pointsData} heatmapData={heatmapData2} {config} />
  <p>
    Most interesting problems in machine learning are nonlinear. Computer vision
    for example is highly nonlinear. Linear and logistic regression are
    therefore not sufficient and we require the use of artificial neural
    networks.
  </p>
  <div class="separator" />
  <div class="info">
    2. What components should a neural network have to solve nonlinear problems?
  </div>
  <p>
    A neural network should use nonlinear activation functions in order to solve
    nonlinear problems. If for example we used an identity function as our
    activation function, no matter how many layers our neural network would
    have, we would only be able to solve linear problems. A sigmoid activation
    function <Latex>{String.raw`\dfrac{1}{1+e^{-x}}`}</Latex> is nonlinear and is
    going to be used as an example in this lecture due to the familiarity from the
    logistic regression lecture . That being said, there are many more nonlinear
    activation functions, which often provide much better properties than the sigmoid
    activation. Those activation functions are going to be a topic of a future lecture.
  </p>
  <p>
    As you have probably already guessed, a nonlinear activation function by
    itself is not sufficient to solve nonlinear problems. The <Highlight
      >linear</Highlight
    > decision boundary that is generated by logistic regression is proof for that.
    To solve a nonlinear problem we need a neural network with at least 1 hidden
    layer.
  </p>
  <p>
    The below architecture of a neural network with two inputs, two hidden
    layers, one output and the sigmoid activation function should be sufficient
    to learn to solve the circular problem above.
  </p>
  <NeuralNetwork />
</Container>
<Footer {notes} />
