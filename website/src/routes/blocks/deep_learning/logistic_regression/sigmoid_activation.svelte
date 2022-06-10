<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Plot from "$lib/Plot.svelte";

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

  let data = [];
  for (let i = -2; i <= 12; i++) {
    let x = i;
    let y = 1;
    if (x <= 5) {
      y = 0;
    }
    data.push({ x, y });

    // needed to make the threshhold absolute vertical
    if (x == 5) {
      data.push({ x: 5, y: 1 });
    }
  }

  let logisticData = [];
  for (let i = -6; i <= 6; i += 0.5) {
    let x = i;
    let y = 1 / (1 + Math.exp(-x));
    logisticData.push({ x, y });
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Sigmoid Activation Function</title>
  <meta
    name="description"
    content="The sigmoid activation function bounds the outputs between 0 and 1, allowing the results to be interpreted as probabilities."
  />
</svelte:head>

<Container>
  <h1>Sigmoid Activation Function</h1>
  <div class="separator" />
  <p>
    Let us start from the basic assumption, that we want to come up with a
    classification algorithm and that we have to deal with only two categories,
    for example cats and dogs.
  </p>
  <div class="separator" />

  <h2>Linear Regression</h2>
  <p>
    We might start by implementing simple linear regression without any
    adjustments. The output <Latex>y</Latex> is either 0 or 1 and we therefore need
    to train a weight <Latex>w</Latex> and a bias <Latex>b</Latex> that produces
    values between 0 and 1. These values can be regarded as probabilities for particular
    category.
  </p>
  <Plot
    pointsData={data.slice(2, data.length - 2)}
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
    We could draw a line just like the one above and at first glance this seems
    to be a reasonable approach. Higher values of some feature correspond to a
    higher probability to belong to the "blue" category and lower values of the
    same feature correspond to a lower probability.
  </p>
  <p>
    Yet we could also get into trouble and our regression line could produce
    results that are above 1 or below 0. This is undesireble, because we need to
    somehow squish our output between 0 and 1 to represent probabilities. If we
    use the same regression line as above but introduce new unseen data our
    model would break apart.
  </p>
  <Plot
    pointsData={data}
    pathsData={[
      { x: 0, y: 0 },
      { x: 10, y: 1 },
    ]}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -2,
      maxX: 12,
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
      numTicks: 8,
    }}
  />
  <div class="separator" />

  <h2>Threshold Activation</h2>
  <p>
    In our second attempt to construct a classification algorithm we will use
    the original threshold activation function that was used in the McCulloch
    and Pitts neuron.
  </p>
  <Plot
    pathsData={data}
    pointsData={regressionData}
    config={{
      minX: -2,
      maxX: 12,
      minY: 0,
      maxY: 1,
      xLabel: "Feature",
      yLabel: "Label",
    }}
  />
  <p>
    The rule in the example above states, that each sample with a feature value
    above 5 is classified into the "blue" category and into the "red" category
    otherwise.
  </p>
  <p>
    This rule perfectly separates the data into the two categories, but now we a
    different type of a problem. How do we find the desired weight <Latex
      >w</Latex
    > and the bias <Latex>b</Latex>. In linear regression we used gradient
    descent, but for that our function needs to be differentiable, which is not
    the case for a threshold function.
  </p>
  <div class="separator" />

  <h2>Sigmoid</h2>
  <p>
    The sigmoid function <Latex>{String.raw`y = \dfrac{1}{1 + e^{-x}}`}</Latex> was
    designed to solve the problems that we faced with the two approaches above.
  </p>

  <Plot
    pathsData={logisticData}
    config={{
      minX: -6,
      maxX: 6,
      minY: 0,
      maxY: 1,
      xLabel: "x",
      yLabel: "y",
    }}
  />
  <p>
    The outputs <Latex>y</Latex> are always bounded between 0 and 1. This allows
    us to interpret the results probabilities. No matter how large or how negative
    the values are, the output is always between 0 and 1.
  </p>
  <p>
    Additioanlly the sigmoid is a softer version of the threshold function. It
    does not have any corners, which allows us to use gradient descent to learn
    the weights and biases.
  </p>
  <div class="separator" />
</Container>
