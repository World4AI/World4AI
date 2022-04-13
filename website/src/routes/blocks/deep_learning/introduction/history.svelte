<script>
  import Question from "$lib/Question.svelte";
  import BiologicalNeuron from "./_history/BiologicalNeuron.svelte";
  import StepFunction from "./_history/StepFunction.svelte";
  import NeuronScaling from "./_history/NeuronScaling.svelte";
  import NeuronScalingAddition from "./_history/NeuronScalingAddition.svelte";
  import NeuronScalingAdditionActivation from "./_history/NeuronScalingAdditionActivation.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Scatterplot from "$lib/Scatterplot.svelte";
  import Table from "$lib/Table.svelte";

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
</script>

<h1>The History of Deep Learning</h1>
<Question>What is the history of deep learning?</Question>
<div class="separator" />

<p>
  In this chapter we will attempt to cover the history of deep learning. In all
  likelihood we will fail at covering all the aspects that lead to the
  development of modern deep learning, so consider this chapter merely a
  starting point.
</p>

<p>
  The history of artificial intelligence is characterized by waves of high hopes
  and high investment followed by disillusionment due to unmet promises and the
  drying up of funding. Neural networks (and by extension deep learning) was
  present in all those waves and the so called AI winters, where there was no
  research.
</p>
<div class="separator" />

<h2>First Wave: Birth of Artificial Neural Networks</h2>
<h3>McCulloch-Pitts Neuron 1943</h3>
<p>
  The first artificial neuron was developed by Warren McCulloch and Walter Pitts<sup
    >1</sup
  >. They tried to come up with a mathematical model of the brain that would be
  able to simulate a biological neuron. From what was known at a time about the
  human brain, the scientists extracted the following information.
</p>
<p>
  The biological neuron receives inputs in a form of a chemical signal. Each
  signal can vary in strength based on the strength of the connection. Those
  signals are aggregated in the center of the neuron. If the cumulative strength
  of all inputs is larger than some threshold the signal travels over the axon
  (the long tail of the neuron) to the connected neurons. In that case we also
  say that the neuron is active.
</p>

<BiologicalNeuron />
<p>
  Let us assume for a second that we are in the position of McCulloch and Pitts
  and would like to emulate the above described behaviour. What would be the key
  characteristics of an artificial neuron?
</p>
<p>
  For once the neuron needs the ability to receive inputs and those inputs need
  to somehow vary in strength. As we are dealing with a mathematical simulation,
  the inputs can be only numbers. In the examples below we will assume that all
  the inputs amount to 1 in order to make the illustrations more intuitive. The
  strength of the signal can be changed with the help of a scaling factor,
  called weight. We therefore adjust the strength of the input by multiplying
  the input <Latex>X</Latex> with a corresponding weight <Latex>W</Latex>. If
  the weight is above 1, the input signal is amplified, if the input is between
  0 and 1 the input signal is dampened. Additionally the strength of the signal
  can be reversed by multiplying the signal with a negative number.
</p>
<p>
  Below is a simple interactive example that allows you to vary the weight and
  observe how the output signal changes. The color of the output signal is blue
  when the output signal is positive and red when it becomes negative. The
  stroke width of the signal output depends on the size of the weight.
</p>
<!--Input weighing svg -->
<NeuronScaling />
<p>
  In the next step we need to figure out the behaviour of the neuron when we
  deal with multiple inputs. Similar to the biological neuron, the input signals
  are accumulated. In the the mathematical model of McCulloch and Pitts the
  weighted inputs are summed up.
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
  A final component that we are missing is the ability of being active based on
  the accumulated input strength. For that McCulloch and Pitts used a simple
  step function. If the weighted sum of the inputs is above a threshold <Latex
    >\theta</Latex
  > the output is 1, else the output is 0. A function that takes the weighted sum
  as input and determines the activation status of a neuron is commonly refered to
  as the <Highlight>activation function</Highlight>.
</p>
<p>
  Below is a step function with a <Latex>\theta</Latex> of 0. You can move the slider
  to observe how the shape of the step function changes due to a different <Latex
    >\theta</Latex
  >.
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
  accumulation of the input signals through a sum and the ability to declare the
  neuron as active or inactive through an activation function is still at the
  core of modern deep learning. The ideas developed by McCulloch and Pitts stood
  the test of time.
</p>
<div class="separator" />

<h3>Perceptron 1957</h3>
<p>
  McCulloch and Pitts provided an architecture for artificial neural networks
  that is still used today. Yet they did not provide a way for a neuron to
  learn.
</p>
<p class="info">
  Learning in machine learning means changing weights <Latex>W</Latex> and the bias
  <Latex>b</Latex>, such that the neuron gets better and better at a particular
  task.
</p>
<p>
  The perceptron developed by Frank Rosenblatt<sup>2</sup> builds upon the idea of
  McCulloch and Pitts and adds a learning rule, that allows us to use an artificial
  neuron in classification tasks.
</p>
<p>
  Imagine we have a labeled dataset with two features and two possible classes,
  as indicated in the scatterplot below.
</p>
<Scatterplot {data} minX={0} maxX={20} minY={0} maxY={15} />
<p>
  It is a relatively easy task for a human being to separate the colored circles
  into the two categories. All we have to do is to draw a line that perfectly
  separates the two groups.
</p>
<Scatterplot
  {data}
  minX={0}
  maxX={20}
  minY={0}
  maxY={15}
  x1Line={0}
  y1Line={0}
  x2Line={22}
  y2Line={15}
/>
<p>
  The perceptron algorithm is designed to find such a line in an automated way.
  In machine learning lingo we also call such a line a <Highlight
    >decision boundary</Highlight
  >. This is due to the fact that the line allows us to make decisions regarding
  the categories of a data point based on the features. Imagine for example we
  encounter a new data point without a label where feature nr. 1 and feature nr.
  2 both correspond to 12. The data point would lie above the decision boundary
  and would therefore be classified as an "orange" class.
</p>
<div class="separator" />

<h3>"Perceptrons" 1969</h3>
<p>
  The McCulloch and Pitts neuron can be used to simulate logical gates, that are
  commonly used in comuter architectures. The idea was that these logical gates
  can be used as buidling block to simulate a human brain.
</p>

<p>
  The <Highlight>or</Highlight> gate produces an output of 1 if either input 1 <Highlight
    >or</Highlight
  >
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
  x1Line={0}
  y1Line={0.8}
  x2Line={0.9}
  y2Line={0}
/>
<p>
  The <Highlight>and</Highlight> gate on the other hand produces an output of 1 when
  input 1 <Highlight>and</Highlight>
  input 2 amount to 1 respectively.
</p>
<p>The decision boundary is easily implemented.</p>
<Table data={andTableData} header={andTableHeader} />

<Scatterplot
  data={andData}
  xLabel="Input 1"
  yLabel="Input 2"
  x1Line={0.2}
  y1Line={1}
  x2Line={1}
  y2Line={0.2}
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
  If you try to separate the data by drawing a single line, you will come to the
  conclusion, that it is impossible.
</p>
<Scatterplot data={xorData} xLabel="Input 1" yLabel="Input 2" />
<p>
  Yet you can separate the data by using a hidden layer. Essentially you combine
  the output from the or gate with the output from the and gate and use those
  outputs in the neuron of the next layer as an input.
</p>
<Table data={mlpTableData} header={mlpTableHeader} />
<p>That makes the separable with a single line.</p>
<Scatterplot
  x1Line={0.2}
  y1Line={0}
  x2Line={1}
  y2Line={0.5}
  data={mlpData}
  xLabel="OR Output"
  yLabel="AND Output"
/>
<div class="separator" />

<h2>First AI Winter</h2>
<p>
  It is not easy to pin down the exact date of the beginning and the end of the
  first AI winter. Generally the book by Minsky and Papert is considered to have
  initiated the decline of research in the field of artificial neural networks.
  Even thogh it was known at the time, that multilayer perceptrons are able to
  solve the xor problem, the general disillusionment and underdelivery of
  artificial intelligence reduced the funding dramatically. Roughly speaking the
  winter held from the beginning of 1970's to the beginning fo 1980's.
</p>
<div class="separator" />

<h2>Second Wave: Neural Networks in the Golden Age of Expert Systems</h2>
<p>
  The second wave of artificial intelligence research that started in the early
  1980's was more favourable towards symbolic artificial intelligence. Symbolic
  AI lead to so called expert systems, where experts input their domain
  knowledge into the program, in the hope of being able to solve intelligence
  though logic and symbols.
</p>
<p>
  Nevertheless there were some brave souls who despite all the criticism of
  neural networks were able to endure and innovate. The groundbreaking research
  that was done in the second wave of artificial intelligence enabled the
  success and recognition that would only come in the third wave of AI.
</p>

<h3>Backpropagation</h3>
<p>Rumelhard and Hinton</p>

<h3>Recurrent Neural Networks</h3>
<p>Hopfield Network</p>
<p>Vanishing Gradient and LSTM</p>

<h3>Convolutional Neural Networks 1989</h3>
<p>Neocognitron by Kunihiko Fukushima</p>

<h3>NeurIPS</h3>

<h2>Third Wave: Modern Deep Learning 2012</h2>
<p>Algorithms, Data and Compute</p>
<h3>ImageNet</h3>
<h3>GPU and AlexNet</h3>
<h3>Turing Award</h3>

<h2>The future is now</h2>
<p>And the rest is history</p>

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
    storage and organization in the brain. Psychological Review Vol. 65, No. 6,
    1958.
  </p>
  <p>
    [3] Minsky M. and Papert S. A. Perceptrons: An Introduction to Computational
    Geometry. MIT Press. 1969
  </p>
</div>

<style>
  .notes p {
    line-height: 1.2;
    font-size: 15px;
    opacity: 80%;
    margin-bottom: 5px;
  }
</style>
