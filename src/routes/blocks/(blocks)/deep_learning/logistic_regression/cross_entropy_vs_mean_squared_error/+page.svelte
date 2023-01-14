<script>
  import Contaienr from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Legend from "$lib/plt/Legend.svelte";

  // table library
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";

  //Difference in Gradients Demonstration
  let graphDataLoss = [];
  let mseData0 = [];
  let mseData1 = [];
  let ceData0 = [];
  let ceData1 = [];

  let graphGradLoss = [];
  let mseGrad0 = [];
  let mseGrad1 = [];
  let ceGrad0 = [];
  let ceGrad1 = [];

  for (let i = 0; i <= 1; i += 0.01) {
    let dataPoint;
    let gradPoint;
    let grad;
    //mse
    // target is 0
    let mse;
    mse = (0 - i) ** 2;
    grad = -2 * (0 - i);
    dataPoint = { x: i, y: mse };
    gradPoint = { x: i, y: grad };
    mseData0.push(dataPoint);
    mseGrad0.push(gradPoint);

    //target is 1
    mse = (1 - i) ** 2;
    grad = -2 * (1 - i);
    dataPoint = { x: i, y: mse };
    mseData1.push(dataPoint);
    gradPoint = { x: i, y: grad };
    mseGrad1.push(gradPoint);

    //cross-entropy
    let ce;
    if (i !== 0 && i !== 1) {
      // target is 0
      ce = -Math.log(1 - i);
      grad = 1 / (1 - i);
      dataPoint = { x: i, y: ce };
      gradPoint = { x: i, y: grad };
      ceData0.push(dataPoint);
      ceGrad0.push(gradPoint);
      // target = 1
      ce = -Math.log(i);
      grad = -1 / i;
      dataPoint = { x: i, y: ce };
      gradPoint = { x: i, y: grad };
      ceData1.push(dataPoint);
      ceGrad1.push(gradPoint);
    }
  }
  graphDataLoss.push(mseData0);
  graphDataLoss.push(mseData1);
  graphDataLoss.push(ceData0);
  graphDataLoss.push(ceData1);

  graphGradLoss.push(mseGrad0);
  graphGradLoss.push(mseGrad1);
  graphGradLoss.push(ceGrad0);
  graphGradLoss.push(ceGrad1);
</script>

<svelte:head>
  <title>Cross-Entropy vs Mean Squared Error - World4AI</title>
  <meta
    name="description"
    content="The cross-entropy loss is more suited for classification tasks, due to higher gradients, especially when the true label and the predicted label diverge strongly. Higher gradients lead to faster convergence. It is therefore advisable to avoid the mean squared error in classification tasks."
  />
</svelte:head>

<Contaienr>
  <h1>Cross-Entropy vs Mean Squared Error</h1>
  <div class="separator" />

  <p>
    The cross-entropy is almost exclusively used as the loss function for
    classification tasks, but it is not obvious why we can not use the mean
    squared error. Actually we can, but as we will see shortly, the
    cross-entropy is a more convenient measure of loss for classification tasks.
  </p>
  <p>
    For this discusson we will deal with a single sample and distinquish between
    different cases.
  </p>

  <Table>
    <TableHead>
      <Row>
        <HeaderEntry>Loss Function</HeaderEntry>
        <HeaderEntry>True Label</HeaderEntry>
        <HeaderEntry>Loss</HeaderEntry>
      </Row>
    </TableHead>
    <TableBody>
      <Row>
        <DataEntry>
          MSE: <Latex>{String.raw`(y-\hat{y})^2`}</Latex>
        </DataEntry>
        <DataEntry>
          <Latex>0</Latex>
        </DataEntry>
        <DataEntry>
          <Latex>{String.raw`(0-\hat{y})^2`}</Latex>
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          MSE: <Latex>{String.raw`(y-\hat{y})^2`}</Latex>
        </DataEntry>
        <DataEntry>
          <Latex>1</Latex>
        </DataEntry>
        <DataEntry>
          <Latex>{String.raw`(1-\hat{y})^2`}</Latex>
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          CE: <Latex
            >{String.raw`-\Big[y \log (\hat{y}) + (1 - y) \log ( 1 - \hat{y})\Big]`}</Latex
          >
        </DataEntry>
        <DataEntry>
          <Latex>0</Latex>
        </DataEntry>
        <DataEntry>
          <Latex>{String.raw`-\log ( 1 - \hat{y})`}</Latex>
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          CE: <Latex
            >{String.raw`-\Big[y \log (\hat{y}) + (1 - y) \log ( 1 - \hat{y})\Big]`}</Latex
          >
        </DataEntry>
        <DataEntry>
          <Latex>1</Latex>
        </DataEntry>
        <DataEntry>
          <Latex>{String.raw`-\log ( \hat{y})`}</Latex>
        </DataEntry>
      </Row>
    </TableBody>
  </Table>
  <p>
    If the label equals to 0 both losses increase as the predicted probability
    grows. If the true label is 1 on the other hand the error decreases when the
    predicted probability grows.
  </p>
  <p>
    Below we plot the mean squared error and the cross-entropy based on the
    predicted probability <Latex>{String.raw`\hat{y}`}</Latex>. The red plot
    depicts the mean squared error, while the blue plot depicts the
    cross-entropy. There are two plots for each of the losses, one for each
    value of the target.
  </p>

  <Plot width={500} height={250} maxWidth={800} domain={[0, 1]} range={[0, 3]}>
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 1, 2, 3]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Predicted Probability" fontSize={15} />
    <YLabel text="Error" fontSize={15} />
    <Path data={graphDataLoss[0]} color="var(--main-color-1)" />
    <Path data={graphDataLoss[1]} color="var(--main-color-1)" />
    <Path data={graphDataLoss[2]} color="var(--main-color-2)" />
    <Path data={graphDataLoss[3]} color="var(--main-color-2)" />
    <Legend
      coordinates={{ x: 0.3, y: 2.8 }}
      legendColor="var(--main-color-2)"
      text="Cross Entropy"
    />
    <Legend
      coordinates={{ x: 0.3, y: 2.5 }}
      legendColor="var(--main-color-1)"
      text="Mean Squared Error"
    />
  </Plot>
  <p>
    The mean squared error and the cross-entropy start at the same position, but
    the difference in errors starts to grow as the predicted probability starts
    to deviate from the true label. The cross-entropy punishes
    misclassifications with a much higher loss, than the mean squared error.
    When we deal with probabilities the difference between the label and the
    predicted probability can not be larger than 1. That means that the mean
    squared error also can not grow beyond 1. The logarithm on the other hand
    literally explodes when the value starts approaching 0.
  </p>
  <p>
    This behaviour can also be observed when we draw the predicted probability
    against the derivative of the loss function. While the derivatives of the
    mean squared error are linear, the cross-entropy derivatives grow
    exponentially when the quality of predictions deteriorates.
  </p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[0, 1]}
    range={[-10, 10]}
  >
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Predicted Probability" fontSize={15} />
    <YLabel text="Derivative" fontSize={15} />
    <Path data={graphGradLoss[0]} color="var(--main-color-1)" />
    <Path data={graphGradLoss[1]} color="var(--main-color-1)" />
    <Path data={graphGradLoss[2]} color="var(--main-color-2)" />
    <Path data={graphGradLoss[3]} color="var(--main-color-2)" />
    <Legend
      coordinates={{ x: 0.3, y: 8 }}
      legendColor="var(--main-color-2)"
      text="Cross Entropy"
    />
    <Legend
      coordinates={{ x: 0.3, y: 6 }}
      legendColor="var(--main-color-1)"
      text="Mean Squared Error"
    />
  </Plot>
  <p>
    The exponential growth of derivative of the cross-entropy loss implies, that
    the gradient descent algorithm will take much larger steps compared to the
    mean squared error, when the classification predictions are way off, thereby
    converging at a higher rate.
  </p>
  <div class="separator" />
</Contaienr>
