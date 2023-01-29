<script>
  import Container from "$lib/Container.svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";

  //Mse simulation
  import Mse from "../_loss/Mse.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";

  let data = [
    { x: 5, y: 20 },
    { x: 10, y: 40 },
    { x: 35, y: 15 },
    { x: 45, y: 59 },
  ];

  let w = 1;
  let b = 0;

  let staticRegressionLine = [];
  let staticLines = [];

  function staticData() {
    //regression line
    let x1 = 0;
    let y1 = b + w * x1;
    let x2 = 60;
    let y2 = b + w * x2;

    staticRegressionLine.push({ x: x1, y: y1 });
    staticRegressionLine.push({ x: x2, y: y2 });

    data.forEach((point) => {
      //lines
      let x1 = point.x;
      let x2 = point.x;
      let y1 = point.y;
      let y2 = b + w * x2;

      let line = [
        { x: x1, y: y1 },
        { x: x2, y: y2 },
      ];
      staticLines.push(line);
    });
  }

  staticData();
</script>

<svelte:head>
  <title>Mean Squared Error - World4AI</title>
  <meta
    name="description"
    content="Linear regression utilizes the mean squared error as a loss function. The goal of linear regression is therefore to minimize the MSE."
  />
</svelte:head>

<Container>
  <h1>Mean Squared Error</h1>
  <div class="separator" />
  <p>
    While we can intuitively tell how far away our line is from some optimal
    location, we need a quantitative measure that the linear regression
    algorithm can use for optimization purposes. In machine learning we use a
    measure called <Highlight>loss function</Highlight> (also called <Highlight
      >error function</Highlight
    > or <Highlight>cost function</Highlight>). The lower the loss function, the
    closer we are to our goal. We can tweak the weights and biases to find a
    minimum value of the loss function.
  </p>
  <p>
    To make our illustrations easier, we will use a dataset with only 4 samples,
    as shown in the illustration below.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={600}
    domain={[0, 60]}
    range={[0, 60]}
    padding={{ top: 40, right: 15, bottom: 50, left: 60 }}
  >
    <Ticks
      xTicks={[0, 10, 20, 30, 40, 50, 60]}
      yTicks={[0, 10, 20, 30, 40, 50, 60]}
      xOffset={-15}
      yOffset={25}
      fontSize={20}
    />
    <XLabel text="Feature" fontSize={25} />
    <YLabel text="Target" fontSize={25} />
    <Circle {data} radius={5} />
  </Plot>
  <p>
    We will start by drawing a 45 degree regression line from the (0,0) to the
    (60, 60) position. While this looks "OK", we do not have a way to compare
    that particular line with any other lines.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={600}
    domain={[0, 60]}
    range={[0, 60]}
    padding={{ top: 40, right: 15, bottom: 50, left: 60 }}
  >
    <Ticks
      xTicks={[0, 10, 20, 30, 40, 50, 60]}
      yTicks={[0, 10, 20, 30, 40, 50, 60]}
      xOffset={-15}
      yOffset={25}
      fontSize={20}
    />
    <XLabel text="Feature" fontSize={25} />
    <YLabel text="Target" fontSize={25} />
    <Circle {data} radius={5} />
    <Path data={staticRegressionLine} />
  </Plot>
  <p>
    The first step is to calculate the error between the actual target
    <Latex>{String.raw`y^{(i)}`}</Latex> and the predicted value <Latex
      >{String.raw`\hat{y}^{(i)}=x^{(i)}w + b`}</Latex
    > for each datapoint
    <Latex>i</Latex>. We can define the difference <Latex
      >{String.raw`y^{(i)} - \hat{y}^{(i)}`}</Latex
    > as the error. Visually we can draw that error as the vertical line that connects
    the regression line with the true target.
  </p>

  <Plot
    width={500}
    height={500}
    maxWidth={600}
    domain={[0, 60]}
    range={[0, 60]}
    padding={{ top: 40, right: 15, bottom: 50, left: 60 }}
  >
    <Ticks
      xTicks={[0, 10, 20, 30, 40, 50, 60]}
      yTicks={[0, 10, 20, 30, 40, 50, 60]}
      xOffset={-15}
      yOffset={25}
      fontSize={20}
    />
    <XLabel text="Feature" fontSize={25} />
    <YLabel text="Target" fontSize={25} />
    {#each staticLines as line}
      <Path data={line} />
    {/each}
    <Circle {data} radius={5} />
    <Path data={staticRegressionLine} />
  </Plot>
  <p>
    Depending on the location and rotation of the regression line <Latex
      >{String.raw`\hat{y} = xw + b`}</Latex
    > and the actual target <Latex>{String.raw`y^{(i)}`}</Latex>, the target
    might be above or below the regression line and thus the error can either be
    positive or negative. If we tried to sum up all the errors in the dataset <Latex
      >{String.raw`\sum_i^n y^{(i)} - \hat{y}^{(i)}`}</Latex
    > the positive and the negative errors would offset each other. If the errors
    were symmetrical above and below the line, we could end up with a summed error
    of 0. This is not what we want. We want positive and negative errors to contribute
    equally to our loss measure, without offsetting each other.
  </p>
  <p>
    The loss that is actually used in linear regression is called the <Highlight
    >
      mean squared error (MSE)</Highlight
    >. The MSE takes each of the individual errors to the power of 2, <Latex
      >{String.raw`(y^{(i)} - \hat{y}^{(i)})^2`}</Latex
    >, to get rid of the negative sign. The average of the errors is defined as
    the mean squared error.
  </p>
  <Alert type="info">
    <Latex
      >{String.raw`MSE=\dfrac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)} )^2`}</Latex
    >
  </Alert>
  <p>Remember that we should try to express all operations in matrix notation in order to make use of parallelization. For the mean squared error that would look as follows.</p>
  <Alert type="info">
    <Latex
      >{String.raw`
\mathbf{\hat{y}} = \mathbf{Xw}^T \\
MSE=mean\Big(\big[\mathbf{y} - \mathbf{\hat{y}}\big]^2\Big) \\
`}</Latex
    >
  </Alert>
  <p>In the first step we calculate many predictions  <Latex>{String.raw`\mathbf{\hat{y}}`}</Latex> simultaneously and calculate the vector of errors <Latex>{String.raw`\mathbf{y - \hat{y}}`}</Latex>. When we square the resulting vector, that implies that each individual unit within the vector is multiplied by itself in parallel. Finally we calculate the mean of the vector. For that purpose deep learning libraries provide a <em>mean()</em> operation, that takes a vector as input and generates a mean from all individual scalars within that vector.</p>
  <p>
    We can visuallize the mean squared error by drawing actual squares. Each
    data point has a corresponding square and the larger the area of that
    square. Try to use the example below and move the weight and the bias.
    Observe how the mean squared error changes based on the parameters.
  </p>
  <Mse {data} {w} {b} />
  <Slider label="Weight" labelId="weight" showValue={true} bind:value={w} min={-20} max={20} step={0.1} />
  <Slider label="Bias" labelId="bias" showValue={true} bind:value={b} min={-50} max={50} />

  <p>
    Different combination of the weight <Latex>w</Latex> and the bias <Latex
      >b</Latex
    > produce different losses and our job is to find the combination that minimizes
    that loss function. Obviously it makes no sense to search manually for those
    parameters. The next section is therefore dedicated to a procedure that is commonly
    used to find the mimimum loss.
  </p>
  <div class="separator" />
</Container>
