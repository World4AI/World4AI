<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";
  import BackpropGraph from "$lib/backprop/BackpropGraph.svelte";

  import { Value, Neuron } from "$lib/Network.js";

  // table library
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";

  import Row from "$lib/base/table/Row.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";

  // Gradient Descent Demonstrations
  let pointsData = [
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

  //the same data, but better suited for training
  let data = [
    { X: [0, 0], y: 0 },
    { X: [0.1, 0.23], y: 0 },
    { X: [0.25, 0.93], y: 0 },
    { X: [0.15, 0.63], y: 0 },
    { X: [0.25, 0.13], y: 0 },
    { X: [0.1, 0.93], y: 0 },
    { X: [0.12, 0.53], y: 0 },
    { X: [0.32, 0.23], y: 0 },
    { X: [0.22, 0.5], y: 0 },
    { X: [0.49, 0.1], y: 0 },
    { X: [0.45, 0.3], y: 0 },
    { X: [0.4, 0.7], y: 0 },
    { X: [0.35, 0.5], y: 0 },
    { X: [0.25, 0.7], y: 0 },
    { X: [0.2, 0.2], y: 0 },
    { X: [1, 1], y: 1 },
    { X: [0.75, 0.89], y: 1 },
    { X: [0.75, 0.75], y: 1 },
    { X: [0.95, 0.7], y: 1 },
    { X: [0.85, 0.7], y: 1 },
    { X: [0.65, 0.8], y: 1 },
    { X: [0.85, 0.4], y: 1 },
    { X: [0.75, 0.25], y: 1 },
    { X: [0.75, 0.55], y: 1 },
    { X: [0.95, 0.35], y: 1 },
    { X: [0.85, 0.15], y: 1 },
    { X: [0.85, 0.95], y: 1 },
    { X: [0.9, 0.55], y: 1 },
    { X: [0.9, 0.28], y: 1 },
    { X: [0.98, 0.95], y: 1 },
  ];
  let ce = new Value(0);
  let crossEntropy = 0;
  let neuron = new Neuron(2, "sigmoid");
  $: w1 = neuron.w[0].data;
  $: w2 = neuron.w[1].data;
  $: b = neuron.b.data;

  let pathsData = [];
  $: {
    let x1 = 0;
    let x2 = 1;

    let y1 = (-b - w1 * x1) / w2;
    let y2 = (-b - w1 * x2) / w2;
    pathsData = [
      { x: x1, y: y1 },
      { x: x2, y: y2 },
    ];
  }

  function train() {
    ce = new Value(0);
    data.forEach((point) => {
      let pred = neuron.forward(point.X);
      if (point.y === 0) {
        let one = new Value(1);
        ce = ce.add(one.sub(pred).log().neg());
      } else if (point.y === 1) {
        ce = ce.add(pred.log().neg());
      }
    });
    crossEntropy = ce.data;
    ce.backward();
    neuron.parameters().forEach((param) => {
      param.data -= 0.01 * param.grad;
    });
    neuron.zeroGrad();
    neuron = neuron;
  }

  // to visualize the construction of the graph
  let step1;
  let step2;
  let step3;
  let step4;

  function steps() {
    let w1 = new Value(0.5);
    let w2 = new Value(-0.5);
    w1._name = "Weight: w_1";
    w2._name = "Weight: w_2";
    let b = new Value(1);
    b._name = "Bias b";
    let x1 = new Value(data[16].X[0]);
    let x2 = new Value(data[16].X[1]);
    x1._name = "Feature 1: x_1";
    x2._name = "Feature 2: x_2";
    let s1 = w1.mul(x1);
    let s2 = w2.mul(x2);
    s1._name = "s_1";
    s2._name = "s_2";
    let z = s1.add(s2).add(b);
    z._name = "z";
    step1 = JSON.parse(JSON.stringify(z));

    let sigmoid = z.sigmoid();
    sigmoid._name = "a";
    step2 = JSON.parse(JSON.stringify(sigmoid));

    let loss = sigmoid.log().neg();
    loss._name = "L";
    step3 = JSON.parse(JSON.stringify(loss));

    loss.backward();
    step4 = JSON.parse(JSON.stringify(loss));
  }
  steps();
</script>

<svelte:head>
  <title>Minimizing Cross-Entropy - World4AI</title>
  <meta
    name="description"
    content="The optimal weights and bias for logistic regression can be obtained by minimizing the cross-entropy loss using the gradient descent algorithm."
  />
</svelte:head>

<h1>Minimizing Cross-Entropy</h1>
<div class="separator" />

<Container>
  <p>
    We finally have the means to find the weight vector <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and the bias
    <Latex>b</Latex> that minimize the binary cross-entropy loss, so let's see how
    we can accomplish this goal.
  </p>
  <Alert type="info">
    <Latex
      >{String.raw`
      \text{Cross-Enropy} = L =  - \dfrac{1}{n} \sum_i \Big[y^{(i)} \log \sigma(z^{(i)}) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \Big] \\
  `}
    </Latex>
  </Alert>
  <p>
    The cross-entropy loss is a relatively complex composite function with many
    moving parts, but if we focus on the atomic componets of the function and
    construct a computational graph, as we did for linear regression, we can
    utilize the chain rule and the calculation of gradients becomes relatively
    straightforward.
  </p>
  <p>
    In this sec1ion we will try to find the optimal weights and bias that define
    the decision boundary between the two categories in the plot below.
  </p>

  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[0, 1]}
    range={[0, 1]}
    padding={{ top: 10, right: 10, bottom: 40, left: 40 }}
  >
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Circle data={pointsData[0]} />
    <Circle data={pointsData[1]} color={"var(--main-color-2)"} />
  </Plot>
  <p>
    The dataset consists of two features, so we start to build our computational
    graph by multiplying each of the features by a corresponding vector.
  </p>
  <Table>
    <TableHead>
      <Row>
        <HeaderEntry><Latex>x_1</Latex></HeaderEntry>
        <HeaderEntry><Latex>x_2</Latex></HeaderEntry>
        <HeaderEntry><Latex>y</Latex></HeaderEntry>
      </Row>
    </TableHead>
    <TableBody>
      <Row>
        <DataEntry>{pointsData[0][0].x}</DataEntry>
        <DataEntry>{pointsData[0][0].y}</DataEntry>
        <DataEntry>{0}</DataEntry>
      </Row>
      <Row>
        <DataEntry>{pointsData[0][1].x}</DataEntry>
        <DataEntry>{pointsData[0][1].y}</DataEntry>
        <DataEntry>{0}</DataEntry>
      </Row>

      <Row>
        <DataEntry>...</DataEntry>
        <DataEntry>...</DataEntry>
        <DataEntry>...</DataEntry>
      </Row>
      <Row>
        <DataEntry>{pointsData[1][1].x}</DataEntry>
        <DataEntry>{pointsData[1][1].y}</DataEntry>
        <DataEntry>{1}</DataEntry>
      </Row>
      <Row>
        <DataEntry>{pointsData[1][0].x}</DataEntry>
        <DataEntry>{pointsData[1][0].y}</DataEntry>
        <DataEntry>{1}</DataEntry>
      </Row>
    </TableBody>
  </Table>
  <p>
    We call the two scaled values <Latex>s_1</Latex> and <Latex>s_2</Latex> respectively
    and add them together.
  </p>
  <Latex>s_1 = w_1 x_1 \\ s_2 = w_1 x_1</Latex>
  <p>
    We sum the two scaled values and add the bias to get the net input <Latex
      >z</Latex
    >.
  </p>
  <Latex>{String.raw`z = s_1 + s_2 + b`}</Latex>
  <p>
    Assuming that the features of the sample are 0.75 and 0.89 respectively, the
    weights are 0.5 and -0.5 respectively and the bias is 1, we get the
    following computational graph so far.
  </p>
  <BackpropGraph graph={step1} maxWidth={450} height={710} width={600} />
  <p>
    In the next step the net input is used as an input into the sigmoid function
    <Latex>{String.raw`\dfrac{1}{1 + e^{-z}}`}</Latex> to get the output <Latex
      >a</Latex
    >.
  </p>
  <BackpropGraph graph={step2} maxWidth={450} height={910} width={600} />
  <p>
    Next we use the output of the sigmoid as an input to the cross-entropy loss.
    When you look at the (single sample) loss function
    <Latex
      >{String.raw`
      L =  -\Big[y \log a + (1 - y) \log(1 - a) \Big]
  `}</Latex
    >
    , you will notice that the loss is dependent on the label <Latex>y</Latex>.
    If the label is 1, the loss collapses to <Latex>-\log (a)</Latex> and if the
    label is 0 the loss collapses to <Latex>-\log(1 - a)</Latex>. The sample
    that we have been looking so far corresponds to a label of 1, so our
    computational graph looks as follows.
  </p>
  <BackpropGraph graph={step3} maxWidth={450} height={1300} width={600} />
  <p>
    When we deal with batch or mini-batch gradient descent, we would do the same
    exercise for many other samples and our computational graph would get
    additional nodes. But let's keep the computation simple and assume that we
    are dealing with stochastic gradient descent and would like to calculate the
    gradients using a single sample. The procedure is obviously the same that we
    used with linear regression. We start at the top node and keep calculating
    the intermediary gradients until we reach the weiths and the bias. Along the
    way we multiply the local gradients by the gradients from the above nodes.
  </p>
  <p />
  <p>
    You should be already familiar with basic differentiation rules. The only
    difficulty you might face is the derivative of the sigmoid function
    <Latex>{String.raw`\sigma(z) = \dfrac{1}{1 + e^{-z}}`}</Latex> with respect to
    the net input <Latex>z</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\dfrac{\partial}{\partial z}\sigma = \sigma(z) (1 - \sigma(z)) `}</Latex
    >
  </div>
  <p>
    The derivative of the sigmoid function is relatively straightforward, but
    the derivation process is somewhat mathematically involved. It is not
    necessary to know the exact steps how we arrive at the derivative, but if
    you are interested, below we provide the necessary steps.
  </p>
  <Alert type="info">
    <Latex
      >{String.raw`
    \begin{aligned}
    \dfrac{\partial}{\partial z}\sigma &= \dfrac{\partial}{\partial z} \dfrac{1}{1 + e^{-z}} \\
    &= \dfrac{\partial}{\partial z} ({1 + e^{-z}})^{-1} \\
    &= -({1 + e^{-z}})^{-2} * -e^{-z} \\
    &= \dfrac{ e^{-z}}{({1 + e^{-z}})^{-2}} \\
    &= \dfrac{1}{{1 + e^{-z}}} \dfrac{ e^{-z}}{{1 + e^{-z}}} \\
    &= \dfrac{1}{{1 + e^{-z}}} \dfrac{ 1 + e^{-z} - 1}{{1 + e^{-z}}} \\
    &= \dfrac{1}{{1 + e^{-z}}} \Big(\dfrac{ 1 + e^{-z}}{1 + e^{-z}} - \dfrac{1}{{1 + e^{-z}}}\Big) \\
    &= \dfrac{1}{{1 + e^{-z}}} \Big(1 - \dfrac{1}{{1 + e^{-z}}}\Big) \\
    &= \sigma(z) (1 - \sigma(z)) 
    \end{aligned}
    `}</Latex
    >
  </Alert>
  <p>
    Calculating the gradients from top to bottom results in the following
    gradients.
  </p>
  <BackpropGraph graph={step4} maxWidth={450} height={1300} width={600} />
  <p>
    When we deal with several samples the computation does not get much more
    complicated.
  </p>
  <Latex
    >{String.raw`
      L =  - \dfrac{1}{n} \sum_i \Big[y^{(i)} \log \sigma(z^{(i)}) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \Big] \\
  `}</Latex
  >
  <p>
    As always the gradient of a sum is the sum of the gradients, so the weights
    and the bias would accumulate the gradients through several samples and
    eventually scaled by <Latex>{String.raw`\dfrac{1}{n}`}</Latex>.
  </p>

  <p>
    In the interactive example below we demonstrate the gradient descent
    algorithm for logistic regression. This is the same example that you tried
    to solve manually in a previous chapter. Start the algorithm and observe how
    the loss decreases over time.
  </p>
  <ButtonContainer>
    <PlayButton f={train} delta={100} />
  </ButtonContainer>
  <Table>
    <TableHead>
      <Row>
        <HeaderEntry>Variable</HeaderEntry>
        <HeaderEntry>Value</HeaderEntry>
      </Row>
    </TableHead>
    <TableBody>
      <Row>
        <DataEntry>
          <Latex>L</Latex>
        </DataEntry>
        <DataEntry>
          {crossEntropy.toFixed(2)}
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          <Latex>w_1</Latex>
        </DataEntry>
        <DataEntry>
          {w1.toFixed(2)}
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          <Latex>w_2</Latex>
        </DataEntry>
        <DataEntry>
          {w2.toFixed(2)}
        </DataEntry>
      </Row>
      <Row>
        <DataEntry>
          <Latex>b</Latex>
        </DataEntry>
        <DataEntry>
          {b.toFixed(2)}
        </DataEntry>
      </Row>
    </TableBody>
  </Table>
  <Plot
    padding={{ top: 10, right: 10, bottom: 40, left: 40 }}
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
    <Path data={pathsData} />
    <Circle data={pointsData[0]} />
    <Circle data={pointsData[1]} color={"var(--main-color-2)"} />
  </Plot>
  <p>
    The gradient descent algorithm learns to separate the data in a matter of
    seconds.
  </p>
  <div class="separator" />
</Container>
