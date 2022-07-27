<script>
  import Container from "$lib/Container.svelte";
  import Table from "$lib/Table.svelte";
  import Latex from "$lib/Latex.svelte";
  // plotting components
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Contour from "$lib/plt/Contour.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Path from "$lib/plt/Path.svelte";

  import * as d3 from "d3";

  //gradient descent
  function calculateGradients(gradFunc) {
    let epochs = 25;
    let coordinates = [];
    coordinates.push({ x: 190, y: 190 });
    for (let i = 0; i < epochs; i++) {
      let x;
      let y;
      x = coordinates[i].x;
      y = coordinates[i].y;
      let grads = gradFunc(x, y);
      x = x - 0.1 * grads.x;
      y = y - 0.1 * grads.y;
      let coordinate = { x, y };
      coordinates.push(coordinate);
    }
    return coordinates;
  }

  let value = (x, y) => x ** 2 + y ** 2;
  let gradient = (x, y) => {
    return { x: 2 * x, y: 2 * y };
  };

  let value2 = (x, y) => x ** 2 + 9 * y ** 2;
  let gradient2 = (x, y) => {
    return { x: 2 * x, y: 18 * y };
  };

  let value3 = (x, y) => x ** 2 + 10 * y ** 2;
  let gradient3 = (x, y) => {
    return { x: 2 * x, y: 20 * y };
  };

  let coordinates = calculateGradients(gradient);
  let coordinates2 = calculateGradients(gradient2);
  let coordinates3 = calculateGradients(gradient3);

  let header = ["Height", "Weight"];
  let data = [
    [1.72, 85],
    [1.92, 92],
    [1.55, 52],
    [1.62, 61],
    [1.7, 71],
  ];

  // normalize and standardize the data
  let minHeight = Number.MAX_VALUE;
  let maxHeight = Number.MIN_VALUE;
  let minWeight = Number.MAX_VALUE;
  let maxWeight = Number.MIN_VALUE;

  let muHeight = 0;
  let sigmaHeight = 0;

  let muWeight = 0;
  let sigmaWeight = 0;

  //calcualte statistics needed for normalization and standardization
  data.forEach((point) => {
    let height = point[0];
    let weight = point[1];

    // calculate min and max
    if (height < minHeight) {
      minHeight = height;
    }

    if (height > maxHeight) {
      maxHeight = height;
    }

    if (weight < minWeight) {
      minWeight = weight;
    }

    if (weight > maxWeight) {
      maxWeight = weight;
    }
  });

  data.forEach((point, idx) => {
    let height = point[0];
    let weight = point[1];

    muHeight = muHeight + (1 / (idx + 1)) * (height - muHeight);
    muWeight = muWeight + (1 / (idx + 1)) * (weight - muWeight);
  });

  data.forEach((point, idx) => {
    let height = point[0];
    let weight = point[1];

    sigmaHeight = sigmaHeight + (height - muHeight) ** 2;
    sigmaWeight = sigmaWeight + (weight - muWeight) ** 2;
  });

  sigmaHeight = Math.sqrt(sigmaHeight / data[0].length);
  sigmaWeight = Math.sqrt(sigmaWeight / data[1].length);

  let normalizedData = [];
  let standardizedData = [];
  data.forEach((point) => {
    let height = point[0];
    let weight = point[1];
    let normalHeight = ((height - minHeight) / (maxHeight - minHeight)).toFixed(
      2
    );
    let normalWeight = ((weight - minWeight) / (maxWeight - minWeight)).toFixed(
      2
    );
    let standardHeight = ((height - muHeight) / sigmaHeight).toFixed(2);
    let standardWeight = ((weight - muWeight) / sigmaWeight).toFixed(2);
    let normalizedPoint = [normalHeight, normalWeight];
    let standardPoint = [standardHeight, standardWeight];
    normalizedData.push(normalizedPoint);
    standardizedData.push(standardPoint);
  });
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Feature Scaling</title>
  <meta
    name="description"
    content="When we encounter features with different scales, there is chance that either training will be slow or that gradient descent will not converge. To avoid those problems we scale input features either though normalization or standardization."
  />
</svelte:head>

<h1>Feature Scaling</h1>
<div class="separator" />
<Container>
  <p>
    You will often face datasets with features that are based on different
    scales. One such dataset could for example contain the height and the weight
    of a person as features.
  </p>
</Container>
<Container maxWidth="500px">
  <Table {header} {data} />
</Container>
<Container>
  <p>
    You have probably already guessed that we used the metric system to depict
    the weight and the height of a person. For height we used meters, which can
    range (roughly) from 0.5 meters to 2.1 meters. For weight we used kilogram,
    which can range from 4kg to 120kg for an average person of different ages.
    If we used these units in training without any rescaling, our neural network
    might either take a very long time to converge or would not converge at all.
  </p>
  <p>
    We can demonstrate this idea by looking at the example below, where we use
    the function <Latex>f(x, y) = x^2 + y^2</Latex> to construct the contour lines.
    This is a bowl shaped function that we observe from above. The different colors
    represent the different values of <Latex>f(x, y)</Latex>. The darker the
    value, the lower the output and the closer we are to the optimium. The
    lowest value is at the point <Latex>(0, 0)</Latex>. The contour lines have a
    circlular shape and are perfectly symmetrical, because both variables <Latex
      >x</Latex
    > and <Latex>y</Latex> have the same impact on the function output. Due to this
    symmetry, it does not matter what the starting x and y values we pick, we will
    move in a straight line towards the minimum.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={700}
    domain={[-200, 200]}
    range={[-200, 200]}
    padding={{ top: 25, right: 25, bottom: 25, left: 25 }}
  >
    <Contour
      f={value}
      thresholds={d3.range(1, 19).map((i) => Math.pow(2, i))}
      scale="log"
    />
    <Path data={coordinates} strokeDashArray={"4 8"} />
    <Circle data={coordinates} radius={2} color={"var(--main-color-3)"} />
    <Ticks
      xTicks={[-200, -150, -100, -50, 0, 50, 100, 150, 200]}
      yTicks={[-200, -150, -100, -50, 0, 50, 100, 150, 200]}
      fontSize={10}
    />
  </Plot>
  <p>
    What happens then if the variables have a non symmetrical impact on the
    output? Let us consider the function <Latex>f(x, y) = x^2 + 9y^2</Latex>. We
    get non symmetrical contour lines and a zigzagging effect. Gradient descent
    does not move the x and y values in a straight line, but oscilates in the y
    direction. When we move in the <Latex>x</Latex> direction, we move by <Latex
      >2x * \alpha</Latex
    >, when we move into y direction, we move by <Latex>18y * \alpha</Latex>, so
    naturally there is a higher chance to overshoot towards <Latex>y</Latex>.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={700}
    domain={[-200, 200]}
    range={[-200, 200]}
    padding={{ top: 25, right: 25, bottom: 25, left: 25 }}
  >
    <Contour
      f={value2}
      thresholds={d3.range(1, 19).map((i) => Math.pow(2, i))}
      scale="log"
    />
    <Path data={coordinates2} strokeDashArray={"4 8"} />
    <Circle data={coordinates2} radius={2} color={"var(--main-color-3)"} />
    <Ticks
      xTicks={[-200, -150, -100, -50, 0, 50, 100, 150, 200]}
      yTicks={[-200, -150, -100, -50, 0, 50, 100, 150, 200]}
      fontSize={10}
    />
  </Plot>
  <p>
    In some cases the value could oscilate indefinetly and never converge to the
    optimum. This is the case for <Latex>f(x, y) = x^2 + 10y^2</Latex> with a learning
    rate of 0.1.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={700}
    domain={[-200, 200]}
    range={[-200, 200]}
    padding={{ top: 25, right: 25, bottom: 25, left: 25 }}
  >
    <Contour
      f={value3}
      thresholds={d3.range(1, 19).map((i) => Math.pow(2, i))}
      scale="log"
    />
    <Path data={coordinates3} strokeDashArray={"4 8"} />
    <Circle data={coordinates3} radius={2} color={"var(--main-color-3)"} />
    <Ticks
      xTicks={[-200, -150, -100, -50, 0, 50, 100, 150, 200]}
      yTicks={[-200, -150, -100, -50, 0, 50, 100, 150, 200]}
      fontSize={10}
    />
  </Plot>
  <p>
    There is a couple of things we can do to reduce the chances of oscilating.
    We could for example use a lower learning rate for both variables, but that
    might slow down the learning process significantly. Or we could use a
    different learning rate for each feature, but tweaking many thousands of
    learning rates seems unfeasable. In practice we scale the input features by
    normalizing or standardizing the inputs. Those techniques bring the features
    on the same scale. It should not matter a lot which of the two procedures
    you employ, just keep in mind to scale your inputs before you start the
    training.
  </p>
  <div class="separator" />

  <h2>Normalization</h2>
  <p>
    Normalization, also called min-max scaling, transorms the features into a
    0-1 range.
  </p>
  <Latex
    >{String.raw`x^{(i)}_j = \dfrac{x_j^{(i)} - \min(x_j)}{\max(x_j) - \min(x_j)}`}</Latex
  >
  <p>
    The largest value of a feature gets assigned a value of 1, the lowest value
    of a feature gets assigned the value of 0 while the rest of the values are
    scaled between 0 and 1.
  </p>
  <p>
    When we apply normalization to the example above, we end up with the
    following feature values.
  </p>
  <Container maxWidth="500px">
    <Table {header} data={normalizedData} />
  </Container>
  <div class="separator" />

  <h2>Standardization</h2>
  <p>
    The standardization procedure produces feature values that have a mean <Latex
      >\mu</Latex
    > of 0 and a standard deviation <Latex>\sigma</Latex> of 1.
  </p>
  <Latex>{String.raw`x^{(i)}_j = \dfrac{x_j^{(i)} - \mu_j}{\sigma_j}`}</Latex>
  <p>
    When we apply standardization to the example above we end up with the
    following features.
  </p>
  <Container maxWidth="500px">
    <Table {header} data={standardizedData} />
  </Container>
  <div class="separator" />
</Container>
