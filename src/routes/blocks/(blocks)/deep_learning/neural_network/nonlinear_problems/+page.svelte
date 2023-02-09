<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";
  import Alert from "$lib/Alert.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Rectangle from "$lib/plt/Rectangle.svelte";

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

  let heatmapData = [[], []];
  for (let i = 0; i <= numbers; i++) {
    for (let j = 0; j <= numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let coordinate = { x, y };
      if (x + y > 1) {
        heatmapData[0].push(coordinate);
      } else {
        heatmapData[1].push(coordinate);
      }
    }
  }

  let heatmapData2 = [[], []];
  for (let i = 0; i <= numbers; i++) {
    for (let j = 0; j <= numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let coordinate = { x, y };
      if ((x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.12) {
        heatmapData2[0].push(coordinate);
      } else {
        heatmapData2[1].push(coordinate);
      }
      heatmapData2.push(coordinate);
    }
  }

  const layers = [
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-gray-300" },
        { value: "x_2", class: "fill-gray-300" },
      ],
    },
    {
      title: "Hidden 1",
      nodes: [
        { value: "a_1", class: "fill-w4ai-yellow" },
        { value: "a_2", class: "fill-w4ai-yellow" },
        { value: "a_3", class: "fill-w4ai-yellow" },
        { value: "a_4", class: "fill-w4ai-yellow" },
      ],
    },
    {
      title: "Output",
      nodes: [{ value: "o_1", class: "fill-w4ai-blue" }],
    },
  ];
</script>

<svelte:head>
  <title>Nonlinear Problems - World4AI</title>
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
    neural networks, but there are several questions we should ask ourselves
    before we move on to the technicalities of neural networks. Let's start with
    the most obvious question.
  </p>
  <Alert type="info">
    Why do we need neural network when we can solve regression tasks using
    linear regression and classification tasks using logistic regression?
  </Alert>
  <p>
    In the example below we have a classification problem with two classes and
    two features. By visually inspecting the dataset, the human brain can
    quickly separate the data by imagining a circle between the two classes.
  </p>
  <Plot width={500} height={500} maxWidth={600} domain={[0, 1]} range={[0, 1]}>
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <Circle data={pointsData[0]} />
    <Circle data={pointsData[1]} color="var(--main-color-2)" />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
  </Plot>
  <p>
    If you go back to the logistic regression lecture, you will remember that
    logistic regression produces a linear decision boundary<InternalLink
      type="note"
      id="1"
    />. While this might be sufficient for some problems, in our case we would
    misclassify approximately half of the data. Logistic regression can only
    produce a linear decision boundary and therefore can only solve linear
    problems. The data below on the other hand clearly depicts a nonlinear
    problem.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={600}
    domain={[0, 1]}
    range={[0, 1]}
    padding={{ top: 10, right: 40, bottom: 45, left: 45 }}
  >
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <Rectangle data={heatmapData[0]} size={9} color="var(--main-color-3)" />
    <Rectangle data={heatmapData[1]} size={9} color="var(--main-color-4)" />
    <Circle data={pointsData[0]} />
    <Circle data={pointsData[1]} color="var(--main-color-2)" />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
  </Plot>
  <p>
    A neural network on the other hand can theoretically generate an adequate
    decision boundary for nonlinear problems.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={600}
    domain={[0, 1]}
    range={[0, 1]}
    padding={{ top: 10, right: 40, bottom: 45, left: 45 }}
  >
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <Rectangle data={heatmapData2[0]} size={9} color="var(--main-color-3)" />
    <Rectangle data={heatmapData2[1]} size={9} color="var(--main-color-4)" />
    <Circle data={pointsData[0]} />
    <Circle data={pointsData[1]} color="var(--main-color-2)" />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
  </Plot>
  <p>
    Most interesting problems in machine learning are nonlinear. Computer vision
    for example is highly nonlinear. Linear and logistic regression are
    therefore not sufficient and we have to utilize artificial neural networks.
  </p>
  <p>From our discussion above the next question follows naturally.</p>
  <Alert type="info">
    What components and properties should a neural network exhibit to solve
    nonlinear problems?
  </Alert>
  <p>
    A neural network must utilize nonlinear activation functions in order to
    solve nonlinear problems. If for example we used an identity function as our
    activation function, no matter how many layers our neural network would
    have, we would only be able to solve linear problems. A sigmoid activation
    function <Latex>{String.raw`\dfrac{1}{1+e^{-z}}`}</Latex> is nonlinear and is
    going to be used as an example in this lecture. That being said, there are many
    more nonlinear activation functions, which often provide much better properties
    than the sigmoid activation. Additional activation functions are going to be
    discussed in a separate lecture.
  </p>
  <p>
    As you have probably already guessed, a nonlinear activation function by
    itself is not sufficient to solve nonlinear problems. Logistic regression
    for example produces a linear decision boundary, even though it is based on
    the sigmoid activation function.
  </p>
  <Alert type="warning">
    To deal with nonlinear problems we need a neural network with at least 1
    hidden layer.
  </Alert>
  <p>
    The below architecture with two inputs, one hidden layer with four neurons
    and the sigmoid activation function will be utilized to learn to solve the
    circular problem above.
  </p>
  <NeuralNetwork {layers} height={150} padding={{ left: 0, right: 10 }} />
  <p>
    How many hidden layers you eventually use and how many neurons are going to
    be used in a particular layer is up to you, but many problems will require a
    a particualar architecture to be solved efficiently.
  </p>
</Container>
<Footer {notes} />
