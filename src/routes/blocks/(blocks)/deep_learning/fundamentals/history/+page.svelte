<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Alert from "$lib/Alert.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import StepFunction from "../_history/StepFunction.svelte";
  import NeuronScaling from "../_history/NeuronScaling.svelte";
  import NeuronScalingAddition from "../_history/NeuronScalingAddition.svelte";
  import NeuronScalingAdditionActivation from "../_history/NeuronScalingAdditionActivation.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import ForwardBackward from "../_history/ForwardBackward.svelte";
  import Cnn from "../_history/Cnn.svelte";
  import Rnn from "../_history/Rnn.svelte";
  import Relu from "../_history/Relu.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";

  // table library
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  //image
  import neuronImg from "../_history/neuron.png";

  const notes = [
    "The idea and the drawings were popularized by Andrew Ng in his deep learning coursera course.",
  ];

  const references = [
    {
      author: "Warren S. McCulloch and Walter Pitts",
      title: "A logical calculus of the ideas immanent in nervous activity",
      journal: "Bulletin of mathematical biophysics",
      year: "1943",
      pages: "115-133",
      volume: "5",
      issue: "",
    },
    {
      author: "Rosenblatt F",
      title:
        "The Perceptron: A probabilistic model for information storage and organization in the brain",
      journal: "Psychological Review",
      year: "1958",
      pages: "386-408",
      volume: "65",
      issue: "6",
    },
    {
      author: "Minsky M. and Papert S. A.",
      title: "Perceptrons: An Introduction to Computational Geometry",
      journal: "MIT Press",
      year: "1969",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author: "Rumelhart D. Hinton G. Williams R",
      title: "Learning representations by back-propagating errors",
      journal: "Nature",
      year: "1986",
      pages: "533-536",
      volume: "323",
      issue: "6088",
    },
    {
      author: "Hopfield, J",
      title:
        "Neural networks and physical systems with emergent collective computational abilities",
      journal: "Proceedings of the National Academy of Sciences",
      year: "1982",
      pages: "2554–2558",
      volume: "79",
      issue: "8",
    },
    {
      author: "Hochreiter S. and Schmidhuber J",
      title: "Long short-term memory",
      journal: "Neural Computation",
      year: "1997",
      pages: "1735-1780",
      volume: "9",
      issue: "8",
    },
    {
      author: "Hubel D. H. and Wiesel T. N.",
      title: "Receptive fields of single neurones in the cat's striate cortex",
      journal: "The Journal of Physiology",
      year: "1959",
      pages: "574-591",
      volume: "124",
      issue: "3",
    },
    {
      author: "Fukushima K",
      title:
        "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position",
      journal: "Biological Cybernetics",
      year: "1980",
      pages: "193-202",
      volume: "36",
      issue: "4",
    },
    {
      author: "LeCun Y. et. al",
      title: "Backpropagation applied to handwritten zip code recognition",
      journal: "Neural Computation",
      year: "1989",
      pages: "541-551",
      volume: "1",
      issue: "4",
    },
    {
      author: "Nair V. and Hinton G",
      title: "Rectified Linear Units Improve Restricted Boltzmann Machines",
      journal:
        "27th International Conference on International Conference on Machine Learning, ICML'10, USA: Omnipress",
      year: "2010",
      pages: "807-814",
      volume: "",
      issue: "",
    },
    {
      author: "Glorot X. Bordes A. and Bengio Y",
      title: "Deep sparse rectifier neural networks",
      journal:
        "International Conference on Artificial Intelligence and Statistics",
      year: "2011",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author: "Deng J. Dong W. Socher R. Li L.-J Li K. and Fei-Fei L",
      title: "ImageNet: A Large-Scale Hierarchical Image Database",
      journal: "IEEE Computer Vision and Pattern Recognition (CVPR)",
      year: "2009",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author: "Krizhevsky, A. Sutskever I.; Hinton, G. E.",
      title: "ImageNet classification with deep convolutional neural networks",
      journal: "Communications of the ACM",
      year: "2017",
      pages: "84-90",
      volume: "60",
      issue: "6",
    },
  ];

  const category1 = [
    { x: 10, y: 12.04 },
    { x: 8, y: 6.95 },
    { x: 2, y: 9.58 },
    { x: 9, y: 8.81 },
    { x: 4, y: 8.33 },
    { x: 7, y: 9.96 },
    { x: 6, y: 7.24 },
    { x: 4, y: 4.26 },
    { x: 5, y: 12.84 },
    { x: 2, y: 4.82 },
    { x: 1, y: 5.68 },
    { x: 9, y: 9.9 },
    { x: 7, y: 8.2 },
    { x: 6, y: 7.3 },
  ];
  const category2 = [
    { x: 10, y: 2.04 },
    { x: 18, y: 6.95 },
    { x: 13, y: 7.58 },
    { x: 19, y: 2.81 },
    { x: 11, y: 6.33 },
    { x: 14, y: 4.96 },
    { x: 16, y: 7.24 },
    { x: 14, y: 4.26 },
    { x: 12, y: 6.84 },
    { x: 17, y: 4.82 },
    { x: 15, y: 5.68 },
  ];

  let andTableData = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
  ];

  let andTableHeader = ["Input 1", "Input 2", "Output"];

  let orTableData = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
  ];

  let orTableHeader = ["Input 1", "Input 2", "Output"];

  let xorTableData = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
  ];

  let xorTableHeader = ["Input 1", "Input 2", "Output"];

  let mlpTableData = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 0],
  ];
  let mlpTableHeader = ["Input 1", "Input 2", "OR Output", "AND Output", "XOR"];

  // data to show performance of dl dependent on amount of data
  let commonAlgorithms = [];
  for (let i = 0; i < 100; i++) {
    let x = i;
    let y = Math.log10(x + 1);
    let point = { x, y };
    commonAlgorithms.push(point);
  }

  let dlAlgorithms = [];
  for (let i = 0; i < 100; i++) {
    let x = i;
    let y = Math.log10(x + 1) + i * 0.04;
    let point = { x, y };
    dlAlgorithms.push(point);
  }
</script>

<svelte:head>
  <title>Deep Learning History - World4AI</title>
  <meta
    name="description"
    content="The history of deep learning is characterized by two alternating waves: a wave of high investment and so called AI winters. At the moment we face the third AI spring and we might not encounter a third winter."
  />
</svelte:head>

<h1>The History of Deep Learning</h1>
<div class="separator" />
<Container>
  <p>
    The history of artificial intelligence is marked by two alternating waves:
    one of hype and one of pessimism. A wave charecterized by high hopes with
    great funding, commonly known as <Highlight>AI spring</Highlight> , is usually
    followed by disillusionment due to unmet promises and the drying up of funding,
    also known as <Highlight>AI winter</Highlight>.
  </p>
  <p>
    In this section we will attempt to cover those waves. In all likelihood we
    will fail at covering all the aspects that lead to the development of modern
    deep learning, so consider this chapter merely a starting point.
  </p>
  <div class="separator" />

  <h2>First Wave: Birth of Artificial Neural Networks</h2>
  <h3>McCulloch-Pitts Neuron</h3>
  <p>
    Artificial neurons are not exact copies of biological neurons, but we could
    say that some components were borrowed during the development of artificial
    neural networks. Our explanations of biological neurons do not fully
    represent the messy reality, but they are good enough to get us started.
  </p>
  <figure class="flex flex-col justify-center items-center">
    <img src={neuronImg} class="rounded-xl" alt="stylized neuron" />
    <figcaption class="text-sm text-center">
      A stylized biological neuron. <em
        >Made with Stable Diffusion 2.1 using DreamStudio.</em
      >
    </figcaption>
  </figure>
  <p>
    A biological neuron receives inputs through branch-like structures called
    dendrites which are usually connected to other neurons. The strenghts of
    signals varies based on the strength of the connections between the neurons.
    The stronger the connection, the stronger the signal. In the second step the
    incoming signals are aggregated in the center (the nucleus) of the neuron.
    If the cumulative strength of all inputs is larger than some threshold, the
    signal travels over the axon (the long tail of the neuron) to the connected
    neurons. In that case we also say that the neuron is active.
  </p>
  <p>
    Warren McCulloch and Walter Pitts <InternalLink type="reference" id={1} /> tried
    to come up with a mathematical model of the brain that would be able to simulate
    a biological neuron and thus developed the first artificial neuron. Let us assume
    for a second that we are in the position of McCulloch and Pitts and would like
    to emulate the above described behaviour. What would be the key ingredients of
    an artificial neuron?
  </p>
  <p>
    For once the neuron needs to to receive inputs and to adjust their strength.
    As we are dealing with a mathematical model, the inputs must be numerical.
    In the examples below we will assume that all the inputs amount to 1 in
    order to make the illustrations more intuitive, but generally speaking the
    inputs can assume any numerical form. The strength of the signal can be
    changed with the help of a scaling factor, called <Highlight
      >weight</Highlight
    >. We can adjust the strength of the input by multiplying the input <Latex
      >x</Latex
    > with a corresponding weight <Latex>w</Latex>. If the weight is above 1,
    the input signal is amplified, if the weight is between 0 and 1 the input
    signal is dampened. The strength of the signal can also be reversed by
    multiplying the signal with a negative weight.
  </p>
  <Alert type="info">
    Inside the artificial neuron the strength of the input signal <Latex
      >x</Latex
    > is adjusted with the help of the weight <Latex>w</Latex> by multiplying the
    input by the weight: <Latex>x*w</Latex>.
  </Alert>
  <p>
    Below is a simple interactive example that allows you to vary the weight and
    to observe how the strength of the signal changes. The color of the signal
    is blue when the scaled signal is positive and red when it becomes negative.
    The stroke width of the signal depends on the size of the weight.
  </p>
  <!--Input weighing svg -->
  <SvgContainer maxWidth={"700px"}>
    <NeuronScaling />
  </SvgContainer>
  <p>
    In the next step we need to figure out the behaviour of the neuron when we
    deal with multiple inputs. Similar to the biological neuron the input
    signals need to be scaled individually and accumulated. In the mathematical
    model of McCulloch and Pitts the weighted inputs are simply added together.
  </p>
  <p>
    Multiplying each input <Latex>x_j</Latex> by a corresponding weight <Latex
      >w_j</Latex
    > and calculating the sum out of the products is called a <Highlight
      >weighted sum</Highlight
    >. Mathematically we can express this idea as
    <Latex>{String.raw`\sum_j x_jw_j`}</Latex>, where <Latex>j</Latex> is the index
    of the input and the corresponding scaling factor.
  </p>
  <p>
    In the interactive example below, you can vary the weights of two inputs and
    observe how the weighted signals are accumulated.
  </p>
  <!-- this animation demonstrates signal addition in McCulloch Pitts -->
  <SvgContainer maxWidth={"700px"}>
    <NeuronScalingAddition />
  </SvgContainer>
  <p>
    The final component that our artificial neuron needs, is the ability to
    become active based on the accumulated input strength.
  </p>
  <Alert type="info">
    A function that takes the weighted sum as input and determines the
    activation status of a neuron is commonly refered to as the <Highlight
      >activation function</Highlight
    >.
  </Alert>
  <p>
    McCulloch and Pitts used a simple step function as the activation function.
    If the weighted sum of the inputs is above a threshold <Latex>\theta</Latex>
    the output is 1, else the output is 0.
  </p>
  <Latex
    >{String.raw`
      f(\mathbf{w}) = 
      \left\{ 
      \begin{array}{rcl}
      0 & for & \sum_j x_j w_j \leq \theta \\ 
      1 & for & \sum_j x_j w_j > \theta \\
      \end{array}
      \right.
    `}</Latex
  >
  <p>
    Below is an interactive example of a step function with a <Latex
      >\theta</Latex
    > of 0. You can move the slider to observe how the shape of the step function
    changes due to a different
    <Latex>\theta</Latex>.
  </p>
  <SvgContainer maxWidth={"700px"}>
    <StepFunction />
  </SvgContainer>
  <p>
    The last interactive example allows you to vary two weights <Latex
      >w_1</Latex
    >, <Latex>w_2</Latex> and the threshold <Latex>\theta</Latex>. Observe how
    the flow of data changes based on the weights and the threshold.
  </p>
  <SvgContainer maxWidth={"700px"}>
    <NeuronScalingAdditionActivation />
  </SvgContainer>
  <p>
    In practice we replace the threshold <Latex>\theta</Latex> by a so called bias
    <Latex>b</Latex>. To illustrate the procedure let us assume that the
    weighted sum corresponds to the threshold.
  </p>
  <Latex>\sum_j x_j w_j = \theta</Latex>
  <p>We can bring <Latex>\theta</Latex> to the other side of the equation.</p>
  <Latex>\sum_j x_j w_j - \theta = 0</Latex>
  <p>
    And define the negative threshold <Latex>\theta</Latex> as the bias <Latex
      >b</Latex
    >.
  </p>
  <Latex>b = -\theta \\ \sum_j x_j w_j + b = 0 \\</Latex>
  <p>Which leads to the following equation for the threshold function.</p>
  <Latex
    >{String.raw`
      f(\mathbf{w}, b) = 
      \left\{ 
      \begin{array}{rcl}
      0 & for & \sum_j x_j w_j + b \leq 0 \\ 
      1 & for & \sum_j x_j w_j + b > 0 \\
      \end{array}
      \right.
    `}</Latex
  >
  <p>
    It might seem like all we do is reformulate the equation, but the idea is
    actually really powerful. We do not assume to know the bias <Latex>b</Latex
    >. Just as the weights <Latex>w_j</Latex> in the equation, the bias is a learnable
    parameter. When we talk about <Latex>\theta</Latex> on the other hand we assume
    to know the threshold.
  </p>
  <p>
    What we discussed so far is a fully fledged artificial neuron. The
    representation of the strength of the signal through a weight <Latex
      >w</Latex
    >, the accumulation of the input signals through summation and the ability
    to declare the neuron as active or inactive through an activation function
    is still at the core of modern deep learning. The ideas developed by
    McCulloch and Pitts stood the test of time.
  </p>
  <h3>Perceptron</h3>
  <p>
    McCulloch and Pitts provided an architecture for an artificial neuron that
    is still used today. Yet they did not provide a way for a neuron to learn.
  </p>
  <Alert type="info">
    Learning in machine learning means changing weights <Latex>w</Latex> and the
    bias
    <Latex>b</Latex>, such that the neuron gets better and better at a
    particular task.
  </Alert>
  <p>
    The perceptron developed by Frank Rosenblatt<InternalLink
      type="reference"
      id={2}
    /> builds upon the idea of McCulloch and Pitts and adds a learning rule, that
    allows us to use an artificial neuron in classification tasks.
  </p>
  <p>
    Imagine we have a labeled dataset with two features and two possible
    classes, as indicated by the colors in the scatterplot below.
  </p>
  <Plot maxWidth={700} domain={[0, 20]} range={[0, 15]}>
    <Ticks
      xTicks={[0, 5, 10, 15, 20]}
      yTicks={[0, 5, 10, 15]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Circle data={category1} />
    <Circle data={category2} color={"var(--main-color-2)"} />
  </Plot>
  <p>
    It is a relatively easy task for a human being to separate the colored
    circles into the two categories. All we have to do is to draw a line that
    perfectly separates the two groups.
  </p>
  <Plot maxWidth={700} domain={[0, 20]} range={[0, 15]}>
    <Ticks
      xTicks={[0, 5, 10, 15, 20]}
      yTicks={[0, 5, 10, 15]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Circle data={category1} />
    <Circle data={category2} color={"var(--main-color-2)"} />
    <Path
      data={[
        { x: 0, y: 0 },
        { x: 20, y: 15 },
      ]}
    />
  </Plot>
  <p>
    The perceptron algorithm is designed to find such a line in an automated
    way. In machine learning lingo we also call such a line a <Highlight
      >decision boundary</Highlight
    >.
  </p>

  <h3>"Perceptrons"</h3>
  <p>
    The McCulloch and Pitts neuron can be used to simulate logical gates, that
    are commonly used in comuter architectures and researchers assumed at the
    time that these logical gates can be used as buidling blocks to simulate a
    human brain.
  </p>
  <p>
    The <Highlight>or</Highlight> gate for example produces an output of 1 if either
    input 1
    <Highlight>or</Highlight>
    input 2 amount to 1.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each orTableHeader as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each orTableData as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>
    We can use the perceptron algorithm to draw a decision boundary between the
    two classes.
  </p>
  <Plot maxWidth={700} domain={[0, 1]} range={[0, 1]}>
    <Ticks xTicks={[0, 1]} yTicks={[0, 1]} xOffset={-15} yOffset={15} />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Circle data={[{ x: 0, y: 0 }]} />
    <Circle
      data={[
        { x: 0, y: 1 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
      ]}
      color={"var(--main-color-2)"}
    />
    <Path
      data={[
        { x: 0, y: 0.8 },
        { x: 0.9, y: 0 },
      ]}
    />
  </Plot>
  <p>
    The <Highlight>and</Highlight> gate on the other hand produces an output of 1
    when input 1 <Highlight>and</Highlight>
    input 2 amount to 1 respectively.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each andTableHeader as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each andTableData as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>The decision boundary is easily implemented.</p>
  <Plot maxWidth={700} domain={[0, 1]} range={[0, 1]}>
    <Ticks xTicks={[0, 1]} yTicks={[0, 1]} xOffset={-15} yOffset={15} />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Circle
      data={[
        { x: 0, y: 0 },
        { x: 0, y: 1 },
        { x: 1, y: 0 },
      ]}
    />
    <Circle data={[{ x: 1, y: 1 }]} color={"var(--main-color-2)"} />
    <Path
      data={[
        { x: 0.2, y: 1 },
        { x: 1, y: 0.2 },
      ]}
    />
  </Plot>
  <p>
    Marvin Minsky and Seymour Papert published a book named "Perceptrons" <InternalLink
      type="reference"
      id={3}
    />
    in the year 1969. In that book they showed that a single perceptron is not able
    to simulate a so called <Highlight>xor</Highlight> gate. The xor gate (exclusive
    or) outputs 1 only when one and only one of the inputs is 1.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each xorTableHeader as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each xorTableData as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>
    If you try to separate the data by drawing a single line, you will come to
    the conclusion, that it is impossible.
  </p>
  <Plot maxWidth={700} domain={[0, 1]} range={[0, 1]}>
    <Ticks xTicks={[0, 1]} yTicks={[0, 1]} xOffset={-15} yOffset={15} />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
    <Circle
      data={[
        { x: 1, y: 1 },
        { x: 0, y: 0 },
      ]}
    />
    <Circle
      data={[
        { x: 0, y: 1 },
        { x: 1, y: 0 },
      ]}
      color={"var(--main-color-2)"}
    />
  </Plot>
  <p>
    Yet you can separate the data by using a hidden layer. Essentially you
    combine the output from the <em>or</em> gate with the output from the
    <em>and</em> gate and use those outputs as inputs in the neuron of the next layer.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each mlpTableHeader as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each mlpTableData as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>That makes the data separable with a single line.</p>
  <Plot maxWidth={700} domain={[0, 1]} range={[0, 1]}>
    <Ticks xTicks={[0, 1]} yTicks={[0, 1]} xOffset={-15} yOffset={15} />
    <XLabel text="OR Output" fontSize={15} />
    <YLabel text="AND Output" fontSize={15} />
    <Circle
      data={[
        { x: 0, y: 0 },
        { x: 1, y: 1 },
      ]}
    />
    <Circle
      data={[
        { x: 1, y: 0 },
        { x: 1, y: 0 },
      ]}
      color={"var(--main-color-2)"}
    />
    <Path
      data={[
        { x: 0.2, y: 0 },
        { x: 1, y: 0.5 },
      ]}
    />
  </Plot>
  <div class="separator" />

  <h2>First AI Winter</h2>
  <p>
    It is not easy to pin down the exact date of the beginning and the end of
    the first AI winter. Often the book by Minsky and Papert is considered to
    have initiated the decline of research in the field of artificial neural
    networks, even thogh it was known at the time, that multilayer perceptrons
    are able to solve the xor problem. More likely the general disillusionment
    with AI systems and the overpromies that were made by the research community
    lead to drastically reduced funding. Roughly speaking the winter held from
    the beginning of 1970's to the beginning fo 1980's.
  </p>
  <div class="separator" />

  <h2>Second Wave: Neural Networks in the Golden Age of Expert Systems</h2>
  <p>
    The second wave of artificial intelligence research, that started in the
    early 1980's, was more favourable towards symbolic artificial intelligence.
    Symbolic AI lead to so called expert systems, where experts infuse their
    domain knowledge into the program in the hope of being able to solve
    intelligence though logic and symbols.
  </p>
  <p>
    Nevertheless there were some brave souls who despite all the criticism of
    neural networks were able to endure and innovate. The groundbreaking
    research that was done in the second wave of artificial intelligence enabled
    the success and recognition that would only come in the third wave of AI.
  </p>

  <h3>Backpropagation</h3>
  <p>
    The perceptron learning algorithm only works for relatively easy problems.
    For a very long time it was not clear how we could train neural networks
    when we are faced with a complex problem (like image classification) and the
    network has hidden layers. In 1986 Rumelhart, Hinton and Williams published
    a paper that described the <Highlight>backpropagation</Highlight> algorithm <InternalLink
      type="reference"
      id={4}
    />. The procedure combined gradient descent, the chain rule and efficient
    computation to form what has become the backbone of modern deep learning.
    The algorithm is said to have been developed many times before 1986, yet the
    1986 paper has popularized the procedure.
  </p>
  <p>
    The backpropagation algorithm will be covered in a dedicated section, but
    let us shortly cover the meaning of the name "backpropagation". This should
    give you some intuition regarding the workings of the algorithm. Essentially
    modern deep learning consists of two steps: <Highlight
      >feedforward</Highlight
    > and
    <Highlight>backpropagation</Highlight>.
  </p>

  <p>
    So far we have only considered the feedforward step. During this step each
    neuron processes its corresponding inputs and its outputs are fed into the
    next layer. Data flows forward from layer to layer until final outputs, the
    predictions of the model are generated.
  </p>
  <ForwardBackward type="forward" />
  <p>
    Once the neural network has produced outputs, they can be compared to the
    actual targets. For example you can compare the predicted house price with
    the actual house price in your training dataset. This allows the neural
    network to compute the error between the prediction and the so called ground
    truth. This error is in turn propagated backwards layer after layer and each
    weight (and bias) is adjusted proportionally to the contribution of that the
    weight to the overall error.
  </p>
  <ForwardBackward type="backward" />
  <p>
    The invention of the backpropagation algorithm was the crucial discovery
    that gave us the means to train neural networks with billions of parameters.
  </p>

  <h3>Recurrent Neural Networks</h3>
  <p>
    The second wave additionally provided us with a lot of research into the
    field of recurrent neural networks like the Hopfield network<InternalLink
      type="reference"
      id={5}
    /> and LSTM<InternalLink type="reference" id="6" />.
  </p>
  <p>
    Unlike feedforward neural networks, a recurrent neural network (RNN) has
    self reference. When we deal with sequential data like text or stock prices,
    the output that is produced from the previous time step in the sequence
    (e.g. first word of the sentence) is used as an additional input for the
    next time step of the sequence (e.g. second word in the sentence).
  </p>
  <Rnn />

  <h3>Convolutional Neural Networks</h3>
  <p>
    The area of computer vision also made great leaps during the second wave.
    The most prominent architecture that was developed during that time are the
    convolutional neural networks (CNN). The development of CNNs a has a rich
    history, that started with research into the visual cortex of cats <InternalLink
      type="reference"
      id={7}
    />, which lead to the development of the first convolutional architecture by
    Kunihiko Fukushima, neocognitron<InternalLink type="reference" id={8} />,
    which in turn lead to the incorporation of backpropagation into the CNN
    architecture by Yann LeCun<InternalLink type="reference" id={9} />.
  </p>
  <Cnn />
  <p>
    In a convolutional neural network we have a sliding window of neurons, that
    focuses on one area of a picure at a time. Unlike fully connected neural
    networks, this architecture takes into account, that nearby pixels in 2d
    space are related. Even today convolutional neural networks produce state of
    the art results in computer vision.
  </p>

  <h3>NeurIPS</h3>
  <p>
    Last but not least, in the year 1986 the conference and workshop on neural
    information processing systems (NeurIPS) was proposed. NeurIPS is a yearly
    machine learning conference that is still held to this day.
  </p>
  <div class="separator" />

  <h2>Second AI Winter</h2>
  <p>
    The expert systems failed to deliver the promised results, which lead to the
    second AI winter. The winter started in the mid 1990's and ended in the year
    2012.
  </p>
  <div class="separator" />

  <h2>Third Wave: Modern Deep Learning</h2>
  <p>
    In hindsight we can say that deep neural networks required at least three
    components to become successful: algorithmic improvements, large amounts of
    data and computational power.
  </p>
  <h3>Algorithms: ReLU</h3>
  <p>
    Many algorithmic advances were made that allowed researchers to improve the
    performance of neural networks, therefore we can only scratch the surface in
    this chapter. The one that seems trivial on the surface, but is actually one
    of the most significant innovations in deep leaning was the introduction of
    the rectified linar unit (ReLU) as an activation function<InternalLink
      type="reference"
      id={10}
    /><InternalLink type="reference" id={11} />.
  </p>
  <p>
    This activation function returns 0 when <Latex
      >{String.raw`\sum_j x_j w_j +b \leq{0}`}</Latex
    > and <Latex>{String.raw`\sum_j x_j w_j + b`}</Latex> otherwise. In other words,
    the activation function retains positive signals, while the function does not
    become active for negative signals.
  </p>
  <SvgContainer maxWidth={"700px"}>
    <Relu />
  </SvgContainer>

  <p>
    Why this type of function is advantageous will be discussed in a dedicated
    section. For know it is sufficient to know, that the backpropagation
    algorithm works extremely well with the ReLU activation function.
  </p>

  <h3>Data: ImageNet</h3>
  <p>
    While most researchers focused only on the algorithmic side of deep
    learning, in the year 2006 Fei-Fei Li began working on collecting images for
    a dataset suitable for large scale vision tasks. That dataset is known as
    ImageNet<InternalLink type="reference" id={12} />.
  </p>
  <p>
    Nowadays we realize what immense role data plays in deep learning and how
    data hungry deep learning algorithms are <InternalLink
      id={1}
      type={"note"}
    />.
  </p>
  <p>
    The performance of classical machine learning algorithms improves when we
    increase the amount of data for training, but after a while the rate of
    improvement is almost flat.
  </p>

  <Plot maxWidth={700} domain={[0, 100]} range={[0, 6]}>
    <Ticks xTicks={[0]} yTicks={[0]} />
    <XLabel text="Amount Of Data" fontSize={15} />
    <YLabel text="ML Performance" fontSize={15} />
    <Path data={commonAlgorithms} />
  </Plot>
  <p>
    Deep learning algorithms on the other hand have a much steeper curve. The
    more data you provide, the better the overall performance of the algorithm.
    Deep learning scales extremely well with the amount of data.
  </p>
  <Plot maxWidth={700} domain={[0, 100]} range={[0, 6]}>
    <Ticks xTicks={[0]} yTicks={[0]} />
    <XLabel text="Amount Of Data" fontSize={15} />
    <YLabel text="DL Performance" fontSize={15} />
    <Path data={dlAlgorithms} />
  </Plot>

  <p>
    When ImageNet became publicly available to researchers, the potential to
    scale an artificial neural network to achieve unprecedented performance came
    into fruition.
  </p>

  <p>
    But therein also lies a weakness of deep learning. Unless you have at least
    several tens of thousands of samples for training, neural networks will not
    shine and you should use some traditional algorithm like decision trees or
    support vector machines. Compared to humans that can utilize their common
    sense to learn new concepts fairly quickly, neural networks can be extremely
    inefficent.
  </p>

  <h3>Computation: GPU</h3>
  <p>
    Graphics processing units (GPU) were developed independently of deep
    learning for the use in computer games. In a way it was a happy accident
    that the same technology that powers the gaming industry is compatible with
    deep learning. Compared to a CPU, a graphics card posesses thousands of
    cores, which enables extreme parallel computations. Graphics cards suddenly
    allowed researchers to train a model that would take months or even years in
    a matter of mere days.
  </p>

  <h3>AlexNet</h3>
  <p>
    As part of the 2012 ImageNet competition Alex Krizhevsky, Ilya Sutskever and
    Geoffrey Hinton created a convolutional neural network called AlexNet<InternalLink
      type="reference"
      id={13}
    />. The neural network beat the competition by a large margin combining
    state of the art deep learning techniques, Nvidia graphics cards for
    computation and the large scale ImageNet dataset. This moment, often called
    the
    <Highlight>ImageNet moment</Highlight>, is regarded as the birthday of
    modern day deep learning.
  </p>
  <div class="separator" />
  <h2>Aftershock</h2>
  <p>
    The impact that AlexNet had on the artificial intelligence community and the
    general population is hard to quantify, but just a decade after the release
    things have changed dramatiacally. It became quickly apparent that AlexNet
    is just the beginning. The amount of research has skyrocketed and provided
    new state of the art results for ImageNet, until the competition was
    declared as solved. New online courses, books, blog post and YouTube videos
    were published on a regular basis, which in turn allowed beginners to get
    immersed in this fascinating topic. AI research companies like DeepMind and
    OpenAI were founded and existing tech companies like Google and FaceBook
    created their own research laboratories. Deep learning frameworks like
    TensorFlow and PyTorch were opensourced, which reduced the cost and time
    needed to conduct research and deploy industry grade models.
  </p>
  <p>
    While it does not look like we are running out of steam yet with our
    innovations in deep learning, many researchers seem to agree that deep
    learning is not our last invention in artificial intelligence. Yet there
    also seems to be a consensus, that no matter how artificial intelligence of
    the future will look like, deep learning will play a major role in our
    develompent of a much more general intelligence. This is a great time to
    learn about deep learning.
  </p>
</Container>
<Footer {references} {notes} />
