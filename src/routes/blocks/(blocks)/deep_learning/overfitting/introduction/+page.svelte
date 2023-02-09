<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Path from "$lib/plt/Path.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";

  let pointsData1 = [
    { x: 0, y: 0.05 },
    { x: 0.1, y: 0.6 },
    { x: 0.21, y: 0.69 },
    { x: 0.31, y: 0.19 },
    { x: 0.41, y: 0.77 },
    { x: 0.51, y: 0.32 },
    { x: 0.61, y: 0.79 },
    { x: 0.69, y: 0.47 },
    { x: 0.81, y: 0.89 },
    { x: 0.92, y: 0.63 },
    { x: 1, y: 0.62 },
  ];

  let pathsData1 = [
    { x: 0, y: 0.3 },
    { x: 1, y: 0.81 },
  ];

  let pointsData2 = [
    { x: 0.04, y: 0.55 },
    { x: 0.25, y: 0.28 },
    { x: 0.54, y: 0.7 },
    { x: 0.75, y: 0.5 },
    { x: 0.95, y: 0.9 },
  ];

  let pointsData3 = [];
  let pathsData3 = [];
  for (let i = -3; i <= 3.1; i += 0.05) {
    let noiseMin = -0.4;
    let noiseMax = 0.4;
    let noise = Math.random() * (noiseMax - noiseMin) + noiseMin;
    let x = i;
    let y = i ** 2;
    let noiseY = y + noise;
    pointsData3.push({ x, y: noiseY });
    pathsData3.push({ x, y });
  }
</script>

<svelte:head>
  <title>Overfitting - World4AI</title>
  <meta
    name="description"
    content="Overfitting and underfitting are two common problems in machine learning. When the model overfits the data, it tracks the training data too closely and does not generalize well to new data. Underfitting on the other hand means that the model is not powerfull enough to fit the data."
  />
</svelte:head>

<h1>Overfitting</h1>
<div class="separator" />
<Container>
  <p>
    Let us assume that we face data with a single feature and we contemplate how
    we are going to fit some model to the data.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[0, 1]}
    range={[0, 1]}
    padding={{ top: 10, left: 40, right: 10, bottom: 40 }}
  >
    <Circle data={pointsData1} radius={3} />
    <Ticks
      xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} y={490} />
    <YLabel text="Target" fontSize={15} />
  </Plot>
  <p>
    There seems to be a linear relationship between the feature and the target,
    even though there is quite a bit of noise. So we might decide to use linear
    regression for our task.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[0, 1]}
    range={[0, 1]}
    padding={{ top: 10, left: 40, right: 10, bottom: 40 }}
  >
    <Circle data={pointsData1} radius={3} />
    <Path data={pathsData1} />
    <Ticks
      xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} y={490} />
    <YLabel text="Target" fontSize={15} />
  </Plot>
  <p>
    The results looks quite reasonable, but we could also train a neural
    network. In that case we would end up with a model that fits the data very
    closely.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[0, 1]}
    range={[0, 1]}
    padding={{ top: 10, left: 40, right: 10, bottom: 40 }}
  >
    <Circle data={pointsData1} radius={3} />
    <Path data={pointsData1} />
    <Ticks
      xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} y={490} />
    <YLabel text="Target" fontSize={15} />
  </Plot>
  <p>
    While linear regression can only model linear relationships, a neural
    network can theoretically assume any imaginable form. When we try to reduce
    the mean squared error, the neural network will inevitably assume a form
    that fits the data as close as possible. Given enough neurons and layers we
    might end up fitting the data perfectly. Yet when you look at both models
    above you might end up thinking that a simple linear model is actually a
    much better fit. The keyword we are looking for is <Highlight
      >generalization</Highlight
    >.
  </p>
  <Alert type="info">
    A model that generalizes well is good at modelling new unforseen data.
  </Alert>
  <p />
  <p>
    When the two models are faced with new data, the neural network will
    underperform, while the simpler linear regression model will do just fine.
    In other words the linear model generalizes much better in this example.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[0, 1]}
    range={[0, 1]}
    padding={{ top: 10, left: 40, right: 10, bottom: 40 }}
  >
    <Circle data={pointsData1} radius={3} />
    <Path data={pathsData1} />
    <Circle data={pointsData2} radius={3} color={"var(--main-color-2)"} />
    <Path data={pointsData1} />
    <Ticks
      xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} y={490} />
    <YLabel text="Target" fontSize={15} />
  </Plot>
  <p>
    This problem when the model fits too closely to the training data and does
    not generalize well to previously unforseen data is called <Highlight
      >overfitting</Highlight
    >. Overfitting is a common problem in deep learning and we are going to
    spend this entire chapter discussing different remedies.
  </p>
  <p>
    As you might imagine <Highlight>underfitting</Highlight> can also be a potential
    problem in deep learning.
  </p>
  <Alert type="info">
    A model that does not have enough expressiveness or parameters to fit to the
    data, will underfit the data.
  </Alert>
  <p>
    In the example below we face data produced by a quadratic function, yet we
    attempt to use linear regression to fit the data. No matter how hard we try,
    even during training we are going to underperform.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-4, 4]}
    range={[0, 11]}
    padding={{ top: 10, left: 40, right: 10, bottom: 40 }}
  >
    <Circle data={pointsData3} radius={3} />
    <Path
      data={[
        { x: -5, y: 0 },
        { x: 5, y: 11 },
      ]}
    />
    <Ticks
      xTicks={[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]}
      yTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} y={490} />
    <YLabel text="Target" fontSize={15} />
  </Plot>
  <p>
    Solving underfitting is usually a much easier problem, because it is often
    sufficient to increase the parameters of the model. We could for example use
    a neural network to train a model that fits the data better. Increasing the
    number of layers and/or the number of neurons usually solves the problem of
    underfitting.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-4, 4]}
    range={[0, 11]}
    padding={{ top: 10, left: 40, right: 10, bottom: 40 }}
  >
    <Circle data={pointsData3} radius={3} />
    <Path data={pathsData3} stroke={3} />
    <Ticks
      xTicks={[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]}
      yTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature" fontSize={15} y={490} />
    <YLabel text="Target" fontSize={15} />
  </Plot>
  <div class="separator" />
</Container>
