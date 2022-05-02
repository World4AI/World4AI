<script>
  import Container from "$lib/Container.svelte";
  import BiologicalNeuron from "./_history/BiologicalNeuron.svelte";
  import StepFunction from "./_history/StepFunction.svelte";
  import NeuronScaling from "./_history/NeuronScaling.svelte";
  import NeuronScalingAddition from "./_history/NeuronScalingAddition.svelte";
  import NeuronScalingAdditionActivation from "./_history/NeuronScalingAdditionActivation.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import ForwardBackward from "./_history/ForwardBackward.svelte";
  import Cnn from "./_history/Cnn.svelte";
  import Rnn from "./_history/Rnn.svelte";
  import Scatterplot from "$lib/Scatterplot.svelte";
  import Table from "$lib/Table.svelte";
  import Relu from "./_history/Relu.svelte";

  const data = [
    [
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
    ],
    [
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
    ],
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

  const orData = [
    [{ x: 0, y: 0 }],
    [
      { x: 0, y: 1 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
    ],
  ];
  const andData = [
    [
      { x: 0, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 0 },
    ],
    [{ x: 1, y: 1 }],
  ];

  const xorData = [
    [
      { x: 1, y: 1 },
      { x: 0, y: 0 },
    ],
    [
      { x: 0, y: 1 },
      { x: 1, y: 0 },
    ],
  ];
  const mlpData = [
    [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ],
    [
      { x: 1, y: 0 },
      { x: 1, y: 0 },
    ],
  ];

  // create circular data
  let circularData = [[], []];
  let radius = [5, 10];
  let centerX = 10;
  let centerY = 10;
  for (let i = 0; i < radius.length; i++) {
    for (let point = 0; point < 200; point++) {
      let angle = 2 * Math.PI * Math.random();
      let r = radius[i];
      let x = r * Math.cos(angle) + centerX;
      let y = r * Math.sin(angle) + centerY;
      circularData[i].push({ x, y });
    }
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | History</title>
  <meta
    name="description"
    content="The history of deep learning is characterized by waves of high investment and so called AI winters."
  />
</svelte:head>

<Container>
  <h1>The History of Deep Learning</h1>
  <div class="separator" />
  <p>
    In this chapter we will attempt to cover the history of deep learning. In
    all likelihood we will fail at covering all the aspects that lead to the
    development of modern deep learning, so consider this chapter merely a
    starting point.
  </p>

  <p>
    The history of artificial intelligence is characterized by waves of high
    hopes and high investment followed by disillusionment due to unmet promises
    and the drying up of funding. Neural networks (and by extension deep
    learning) was present in all those waves and the so called AI winters, where
    there was no research.
  </p>
  <div class="separator" />

  <h2>First Wave: Birth of Artificial Neural Networks</h2>
  <h3>McCulloch-Pitts Neuron</h3>
  <p>
    The first artificial neuron was developed by Warren McCulloch and Walter
    Pitts<sup>1</sup>. They tried to come up with a mathematical model of the
    brain that would be able to simulate a biological neuron. From what was
    known at a time about the human brain, the scientists extracted the
    following information.
  </p>
  <p>
    The biological neuron receives inputs in a form of a chemical signal. Each
    signal can vary in strength based on the strength of the connection. Those
    signals are aggregated in the center of the neuron. If the cumulative
    strength of all inputs is larger than some threshold the signal travels over
    the axon (the long tail of the neuron) to the connected neurons. In that
    case we also say that the neuron is active.
  </p>

  <BiologicalNeuron />
  <p>
    Let us assume for a second that we are in the position of McCulloch and
    Pitts and would like to emulate the above described behaviour. What would be
    the key characteristics of an artificial neuron?
  </p>
  <p>
    For once the neuron needs the ability to receive inputs and those inputs
    need to somehow vary in strength. As we are dealing with a mathematical
    simulation, the inputs can be only numbers. In the examples below we will
    assume that all the inputs amount to 1 in order to make the illustrations
    more intuitive. The strength of the signal can be changed with the help of a
    scaling factor, called weight. We therefore adjust the strength of the input
    by multiplying the input <Latex>X</Latex> with a corresponding weight <Latex
      >W</Latex
    >. If the weight is above 1, the input signal is amplified, if the input is
    between 0 and 1 the input signal is dampened. Additionally the strength of
    the signal can be reversed by multiplying the signal with a negative number.
  </p>
  <p>
    Below is a simple interactive example that allows you to vary the weight and
    observe how the output signal changes. The color of the output signal is
    blue when the output signal is positive and red when it becomes negative.
    The stroke width of the signal output depends on the size of the weight.
  </p>
  <!--Input weighing svg -->
  <NeuronScaling />
  <p>
    In the next step we need to figure out the behaviour of the neuron when we
    deal with multiple inputs. Similar to the biological neuron, the input
    signals are accumulated. In the the mathematical model of McCulloch and
    Pitts the weighted inputs are summed up.
  </p>
  <p>
    Multiplying each input <Latex>X_i</Latex> by a corresponding weight <Latex
      >W_i</Latex
    > and building a sum out of these produces is called a <Highlight
      >weighted sum</Highlight
    >. Mathematically we can express this idea as <Highlight
      ><Latex>{String.raw`\sum_i X_iW_i`}</Latex></Highlight
    >.
  </p>
  <p>
    In the interactive example below, you can vary the weights of two inputs and
    observe how the weighted signals are accumulated.
  </p>
  <!-- this animation demonstrates signal addition in McCulloch Pitts -->
  <NeuronScalingAddition />

  <p>
    A final component that we are missing is the ability of being active based
    on the accumulated input strength. For that McCulloch and Pitts used a
    simple step function. If the weighted sum of the inputs is above a threshold <Latex
      >\theta</Latex
    > the output is 1, else the output is 0. A function that takes the weighted sum
    as input and determines the activation status of a neuron is commonly refered
    to as the <Highlight>activation function</Highlight>.
  </p>
  <p>
    Below is a step function with a <Latex>\theta</Latex> of 0. You can move the
    slider to observe how the shape of the step function changes due to a different
    <Latex>\theta</Latex>.
  </p>
  <StepFunction />
  <p>
    The last interactive example allows you to vary two weights <Latex
      >\theta</Latex
    >. The output is always either 0 or 1.
  </p>
  <NeuronScalingAdditionActivation />
  <p>
    What we discussed so far is the full fledged artificial neuron. The
    representation of the strength of the signal through a weight, the
    accumulation of the input signals through a sum and the ability to declare
    the neuron as active or inactive through an activation function is still at
    the core of modern deep learning. The ideas developed by McCulloch and Pitts
    stood the test of time.
  </p>
  <div class="separator" />

  <h3>Perceptron</h3>
  <p>
    McCulloch and Pitts provided an architecture for artificial neural networks
    that is still used today. Yet they did not provide a way for a neuron to
    learn.
  </p>
  <p class="info">
    Learning in machine learning means changing weights <Latex>W</Latex> and the
    bias
    <Latex>b</Latex>, such that the neuron gets better and better at a
    particular task.
  </p>
  <p>
    The perceptron developed by Frank Rosenblatt<sup>2</sup> builds upon the idea
    of McCulloch and Pitts and adds a learning rule, that allows us to use an artificial
    neuron in classification tasks.
  </p>
  <p>
    Imagine we have a labeled dataset with two features and two possible
    classes, as indicated in the scatterplot below.
  </p>
  <Scatterplot {data} minX={0} maxX={20} minY={0} maxY={15} />
  <p>
    It is a relatively easy task for a human being to separate the colored
    circles into the two categories. All we have to do is to draw a line that
    perfectly separates the two groups.
  </p>
  <Scatterplot
    {data}
    minX={0}
    maxX={20}
    minY={0}
    maxY={15}
    lines={[{ x1: 0, y1: 0, x2: 22, y2: 15 }]}
  />
  <p>
    The perceptron algorithm is designed to find such a line in an automated
    way. In machine learning lingo we also call such a line a <Highlight
      >decision boundary</Highlight
    >. This is due to the fact that the line allows us to make decisions
    regarding the categories of a data point based on the features. Imagine for
    example we encounter a new data point without a label where feature nr. 1
    and feature nr. 2 both correspond to 12. The data point would lie above the
    decision boundary and would therefore be classified as an "orange" class.
  </p>
  <div class="separator" />

  <h3>"Perceptrons"</h3>
  <p>
    The McCulloch and Pitts neuron can be used to simulate logical gates, that
    are commonly used in comuter architectures. The idea was that these logical
    gates can be used as buidling block to simulate a human brain.
  </p>

  <p>
    The <Highlight>or</Highlight> gate produces an output of 1 if either input 1
    <Highlight>or</Highlight>
    input 2 amount to 1.
  </p>
  <Table data={orTableData} header={orTableHeader} />
  <p>
    We can use the perceptron algorithm to draw a decision boundary between the
    two classes.
  </p>
  <Scatterplot
    data={orData}
    xLabel="Input 1"
    yLabel="Input 2"
    lines={[{ x1: 0, y1: 0.8, x2: 0.9, y2: 0 }]}
  />
  <p>
    The <Highlight>and</Highlight> gate on the other hand produces an output of 1
    when input 1 <Highlight>and</Highlight>
    input 2 amount to 1 respectively.
  </p>
  <Table data={andTableData} header={andTableHeader} />
  <p>The decision boundary is easily implemented.</p>
  <Scatterplot
    data={andData}
    xLabel="Input 1"
    yLabel="Input 2"
    lines={[{ x1: 0.2, y1: 1, x2: 1, y2: 0.2 }]}
  />
  <p>
    Marvin Minsky and Seymour Papert published a book named "Perceptrons"<sup
      >3</sup
    >
    in the year 1969. In that book they showed that a single perceptron is not able
    to simulate a so called <Highlight>xor</Highlight> gate. The xor gate (exclusive
    or) outputs 1 only when one and only one of the inputs are active.
  </p>
  <Table data={xorTableData} header={xorTableHeader} />
  <p>
    If you try to separate the data by drawing a single line, you will come to
    the conclusion, that it is impossible.
  </p>
  <Scatterplot data={xorData} xLabel="Input 1" yLabel="Input 2" />
  <p>
    Yet you can separate the data by using a hidden layer. Essentially you
    combine the output from the or gate with the output from the and gate and
    use those outputs in the neuron of the next layer as an input.
  </p>
  <Table data={mlpTableData} header={mlpTableHeader} />
  <p>That makes the data separable with a single line.</p>
  <Scatterplot
    lines={[{ x1: 0.2, y1: 0, x2: 1, y2: 0.5 }]}
    data={mlpData}
    xLabel="OR Output"
    yLabel="AND Output"
  />
  <div class="separator" />

  <h2>First AI Winter</h2>
  <p>
    It is not easy to pin down the exact date of the beginning and the end of
    the first AI winter. Generally the book by Minsky and Papert is considered
    to have initiated the decline of research in the field of artificial neural
    networks. Even thogh it was known at the time, that multilayer perceptrons
    are able to solve the xor problem, the general disillusionment and
    underdelivery of artificial intelligence reduced the funding dramatically.
    Roughly speaking the winter held from the beginning of 1970's to the
    beginning fo 1980's.
  </p>
  <div class="separator" />

  <h2>Second Wave: Neural Networks in the Golden Age of Expert Systems</h2>
  <p>
    The second wave of artificial intelligence research that started in the
    early 1980's was more favourable towards symbolic artificial intelligence.
    Symbolic AI lead to so called expert systems, where experts input their
    domain knowledge into the program, in the hope of being able to solve
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
    While the perceptron learning algorithm allowed to separate data linearly,
    the algorithm breaks apart when faced with nonlinear data like the one
    displayed below.
  </p>
  <Scatterplot
    data={circularData}
    minX={0}
    maxX={20}
    minY={0}
    maxY={20}
    numTicks={5}
  />
  <p>
    For a relatively long time it was not clear how we could train neural
    networks when the data displays nonlinearity and the network has hidden
    layers. In 1986 Rumelhart, Hinton and Williams published a paper that
    described the backpropagation algorithm<sup>4</sup>. The procedure combined
    gradient descent, the chain rule and efficient computation to form what has
    become the backbone of modern deep learning. The algorithm is said to have
    been developed many times before 1986, yet the 1986 paper has popularized
    the procedure.
  </p>
  <p>
    The backpropagation algorithm will be covered in a dedicated section, but
    let us shortly cover the meaning of the name "backpropagation". This should
    give you some intuition regarding the workings of the algorithm. Essentially
    modern machine learning consists of two steps: <Highlight
      >feedforward</Highlight
    > and
    <Highlight>backpropagation</Highlight>.
  </p>

  <p>
    So far we have only considered the feedforward step. During this step each
    neuron processes its corresponding inputs and its outputs are fed into the
    next layer. Data flows forward from layer to layer until final outputs are
    generated.
  </p>
  <ForwardBackward type="forward" />
  <p>
    Once the neural network produces outputs, they can be compared to the actual
    true values. For example you can compare the predicted house price with the
    actual house price in your training dataset. This allows the neural network
    to compute the error between the prediction and the ground truth. This error
    is in turn is propagated backwards layer after layer. Each weight (and bias)
    is adjusted proportionally to the contribution of that the weight to the
    overall error.
  </p>
  <ForwardBackward type="backward" />
  <p>
    The invention of the backpropagation algorithm was the crucial discovery
    that gave us the means to train neural networks with billions of parameters.
  </p>
  <div class="separator" />

  <h3>Recurrent Neural Networks</h3>
  <p>
    The second wave additionally provided us with a lot of research into the
    field of recurrent neural networks like the Hopfield network<sup>5</sup> and
    LSTM<sup>6</sup>.
  </p>
  <p>
    Unlike classical feedforward neural networks, recurrent neural nets (RNN)
    have self reference. That means that when we deal with sequential data like
    text or stock prices the output that is produced for the previous part in
    the sequence (e.g. first word of the sentence) is used as an additional
    input for the next part of the sequence (e.g. second word in the sentence).
  </p>
  <Rnn />
  <p>
    For example when we input the second word in the sentence into the neuron,
    we additionally use the output of the first word from the same neuron. This
    architecture allows us to deal with information where the order of the data
    matters.
  </p>

  <h3>Convolutional Neural Networks</h3>
  <p>
    The area of computer vision also made great leaps during the second wave.
    The most prominent architecture that was developed during that time are the
    convolutional neural networks (CNN). The development of CNNs is essentially
    a culmination of research that started with research into the visual cortex
    of cats<sup>7</sup>, lead to the development of the first convolutional
    architecture by Kunihiko Fukushima, called neocognitron<sup>8</sup> and
    eventually lead to the incorporation of backpropagation into the CNN
    architecture by Yann LeCun<sup>9</sup>.
  </p>
  <Cnn />
  <p>
    In a convolutional neural network we have a sliding window of neurons, that
    focuses on one area of a picure at a time. This architecture allows to
    preserve the two dimensional structure that is important to visual tasks.
    Even today convolutional neural networks produce state of the art results in
    computer vision.
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
    the rectified linar unit (ReLU) as an activation function<sup>10,11</sup>.
  </p>
  <p>
    This activation function returns 0 when <Latex
      >{String.raw`\mathbf{xw}+b\leq{0}`}</Latex
    > and <Latex>{String.raw`\mathbf{xw}+b`}</Latex> otherwise. In other words, the
    activation function retains positive signals, while the function does not become
    active for negative signals.
  </p>
  <Relu />

  <p>
    Why this type of function is advantageous will be discussed in a dedicated
    section. For know it is sufficient to know, that the backpropagation
    algorithm works extremely well with the ReLU activation function.
  </p>

  <h3>Data: ImageNet</h3>
  <p>
    While most researchers focused on the algorithmic side of the deep learning
    revolution, in the year 2006 Fei-Fei Li began working on collecting images
    for a dataset suitable for large scale vision tasks. That dataset, that is
    still used extensively today became known as ImageNet<sup>12</sup>.
  </p>

  <h3>Computation: GPU</h3>
  <p>
    Graphics processing units (GPU) were developed independently for the use in
    computer games. In a way it was a happy accident that the same technology
    that powers the gaming industry is compatible with deep learning. Compared
    to a CPU a graphics card posesses tousands of cores, which enables extreme
    parallel computations.
  </p>

  <h3>AlexNet</h3>
  <p>coming soon ...</p>

  <h3>Turing Award</h3>
  <p>coming soon ...</p>

  <h2>The future is now</h2>
  <p>coming soon ...</p>

  <div class="separator" />
  <h2>Notes</h2>
  <div class="notes">
    <p>
      [1] Warren S. McCulloch and Walter Pitts. A logical calculus of the ideas
      immanent in nervous activity. Bulletin of mathematical biophysics, vol. 5
      (1943), pp. 115–133.
    </p>
    <p>
      [2] Rosenblatt F. The Perceptron: A probabilistic model for information
      storage and organization in the brain. Psychological Review Vol. 65, No.
      6, 1958.
    </p>
    <p>
      [3] Minsky M. and Papert S. A. Perceptrons: An Introduction to
      Computational Geometry. MIT Press. 1969
    </p>
    <p>
      [4] Rumelhart D and Hinton G and Williams R. Learning representations by
      back-propagating errors. (1986a) Nature. 323 (6088): 533–536
    </p>
    <p>
      [5] Hopfield, J. Neural networks and physical systems with emergent
      collective computational abilities. (1982) Proceedings of the National
      Academy of Sciences. 79 (8): 2554–2558
    </p>
    <p>
      [6] Hochreiter S. and Schmidhuber J. Long short-term memory. 1997. Neural
      Computation. 9 (8): 1735–1780.
    </p>
    <p>
      [7] Hubel D. H. and Wiesel T. N. Receptive fields of single neurones in
      the cat's striate cortex. 1959. The Journal of Physiology. 124 (3):
      574–591
    </p>
    <p />
    <p>
      [8] Fukushima K. Neocognitron: A self-organizing neural network model for
      a mechanism of pattern recognition unaffected by shift in position. 1980.
      Biological Cybernetics. 36 (4): 193–202
    </p>
    <p>
      [9] LeCun Y. et. al. Backpropagation applied to handwritten zip code
      recognition. 1989. Neural Computation, 1(4):541-551,
    </p>
    <p>
      [10] Nair V. and Hinton G. Rectified Linear Units Improve Restricted
      Boltzmann Machines. 2010. 27th International Conference on International
      Conference on Machine Learning, ICML'10, USA: Omnipress, pp. 807–814
    </p>
    <p>
      [11] Glorot X and Bordes A. and Bengio Y. Deep sparse rectifier neural
      networks. 2011. International Conference on Artificial Intelligence and
      Statistics.
    </p>
    <p>
      [12] Deng J. and Dong W. and Socher R. and Li L.-J and Li K. and Fei-Fei
      L. 2009. ImageNet: A Large-Scale Hierarchical Image Database. IEEE
      Computer Vision and Pattern Recognition (CVPR).
    </p>
  </div>
</Container>

<style>
  .notes p {
    line-height: 1.2;
    font-size: 15px;
    opacity: 80%;
    margin-bottom: 5px;
  }
</style>
