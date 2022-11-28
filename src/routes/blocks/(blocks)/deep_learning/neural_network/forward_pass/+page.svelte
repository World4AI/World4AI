<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import ForwardPass from "../_forward/ForwardPass.svelte";
  import Latex from "$lib/Latex.svelte";
  import { NeuralNetwork } from "$lib/NeuralNetwork.js";

  let features = [
    [0.5, 0.5],
    [0.95, 0.5],
    [0.75, 0.5],
  ];

  let labels = [[0], [1], [0]];
  const nn = new NeuralNetwork(0.01, [2, 4, 2, 1], features, labels);

  //subscribe to the stores
  const weightsStore = nn.weightsStore;
  const biasesStore = nn.biasesStore;
  const netInputsStore = nn.netInputsStore;
  const activationsStore = nn.activationsStore;

  nn.epoch();
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Forward Pass</title>
  <meta
    name="description"
    content="In the forward pass the neural network takes features as the input and produces estimates of the outputs. These can be used in the backpropagation algorithm."
  />
</svelte:head>

<h1>Forward Pass</h1>
<div class="separator" />
<Container>
  <p>
    In the forward pass the features in the dataset are processed layer by
    layer, where the outputs from the previous layer are used as inputs into the
    current layer. As the name <Highlight>forward pass</Highlight> suggests, information
    flows unidirectionally from the input layer to the output layer.
  </p>
  <p>
    You can use the interactive example of a neural network below to better
    understand the calculation of a neural network. This is the same
    architecture from the previous section that we are going to use to create a
    nonlinear decision boundary. The leftmost (red) numbers are the two features
    that are used as the input into the neural network. The network has two
    hidden layers with four and two neurons respectively. The neurons in the
    same layer use the same inputs, but have individual weights, which allows
    the neural network to learn different representations based on the same
    input. The single output represents the probability to belong to the
    category 1. Each neuron uses the sigmoid as its activation function. By
    selecting one of the neurons you can observe which weights, inputs and bias
    were used in the calculation. The table of the neural network is designed to
    show the exact values and intermediary steps that were used to calculate the
    output of a particular unit.
  </p>
</Container>
<div class="separator" />
<ForwardPass
  weights={$weightsStore}
  biases={$biasesStore}
  {features}
  {labels}
  netInputs={$netInputsStore}
  activations={$activationsStore}
/>
<div class="separator" />
<Container>
  <p>
    While the interactive example is a good way to demonstrate the functionality
    of the neural network, we need a way to formalize these calculations through
    mathematical notation.
  </p>
  <p>
    In the above example we worked with a single sample, but in practice in the
    forward pass we will use a dataset consisting of many samples, a mini batch <Latex
      >{String.raw`\mathbf{X}`}</Latex
    >. As usual <Latex>{String.raw`\mathbf{X}`}</Latex> is an <Latex
      >n \times m</Latex
    > matrix, where <Latex>n</Latex> (rows) is the number of samples and <Latex
      >m</Latex
    > (columns) is the number of input features.
  </p>
  <Latex
    >{String.raw`
  \mathbf{X} =
  \begin{bmatrix}
  x_1^{(1)} & x_1^{(2)} & x_1^{(3)} & \cdots & x_1^{(m)} \\
  x_2^{(1)} & x_2^{(2)} & x_2^{(3)} & \cdots & x_2^{(m)} \\
  x_3^{(1)} & x_3^{(2)} & x_3^{(3)} & \cdots & x_3^{(m)} \\
  \vdots & \vdots & \vdots & \cdots & \vdots \\
  x_n^{(1)} & x_n^{(2)} & x_n^{(3)} & \cdots & x_n^{(m)} \\
  \end{bmatrix}
    `}</Latex
  >
  <p>
    Each neuron in a particular layer has the same set of inputs, but has its
    own set of weights <Latex>{String.raw`\mathbf{w}`}</Latex> and its own bias <Latex
      >b</Latex
    >. For convenience it makes sence to collect the weights in a matrix <Latex
      >{String.raw`\mathbf{W}`}</Latex
    > and the biases in the vector <Latex>{String.raw`\mathbf{b}`}</Latex>. We
    distinguish between layers by using the superscript <Latex
      >{String.raw`<l>`}</Latex
    >
    , where <Latex>l</Latex> represents the number of the layer. The matrix <Latex
      >{String.raw`\mathbf{W}^{<1>}`}</Latex
    > for example contains weights that connect the input features with the features
    in the first hidden layer.
  </p>
  <p>
    The weight matrix <Latex>{String.raw`\mathbf{W}`}</Latex> is a <Latex
      >d \times m</Latex
    > matrix, where <Latex>m</Latex> is the number of features from the previous
    layer (or input features) and <Latex>d</Latex> is the number of neurons in a
    layer, or put differnetly <Latex>d</Latex> is the number of hidden features produced
    by the weight matrix for the next layer <Latex
      >{String.raw`m^{<l>} = d^{<l+1>}`}</Latex
    >.
    <Latex>{String.raw`\mathbf{b}`}</Latex> is a <Latex>d \times 1</Latex> vector
    of biases.
  </p>
  <p>
    We can calculate the net input <Latex>{String.raw`\mathbf{Z}`}</Latex> by multiplying
    the matrix of features <Latex>{String.raw`\mathbf{X}`}</Latex> with the transpose
    of the weights <Latex>{String.raw`\mathbf{W}^T`}</Latex> and adding the weights
    <Latex>{String.raw`\mathbf{b}`}</Latex>. The result is an <Latex
      >n \times d</Latex
    >
    matrix. Finally we apply an activation function (sigmoid in our case) to all
    net inputs, to get the activation values <Latex
      >{String.raw`\mathbf{A} = a(\mathbf{Z})`}</Latex
    > .
  </p>
  <p>
    We can calculate the outputs for each of the layers by applying the same
    procedure over and over again.
  </p>

  <p>For the first layer we get:</p>
  <Latex
    >{String.raw`\mathbf{Z^{<1>}} = \mathbf{X}\mathbf{W^{<1>T}} + \mathbf{b^{<T>}} \\ 
    `}</Latex
  >
  <br />
  <Latex
    >{String.raw`\mathbf{A^{<1>}} = a( \mathbf{Z^{<l>}})
      \\
      `}</Latex
  >
  <p>For all the other layers we get:</p>
  <Latex
    >{String.raw`\mathbf{Z^{<l>}} = \mathbf{X}\mathbf{A^{<l-1>T}} + \mathbf{b^{<T>}} \\ 
    `}</Latex
  >
  <br />
  <Latex
    >{String.raw`\mathbf{A^{<l>}} = a( \mathbf{Z^{<l>}})
      \\
      `}</Latex
  >
  <p>
    The output of the very last layer <Latex>L</Latex> corresponds to the predicted
    values <Latex>{String.raw`\hat{\mathbf{y}}`}</Latex>. In our case<Latex
      >{String.raw`\hat{y}`}</Latex
    > is a scalar value.
  </p>
  <p>
    The neural network is essentially just a collection of matrix
    multiplications and nonlinear activation functions. Each matrix
    multiplication follows by an activation and each activation is followed by a
    matrix multiplication until the last layer is reached.
  </p>
  <p>
    Essentially a neural network is a deeply nested function composition, where
    the output of a function is used as input into the next function.
  </p>
  <Latex
    >{String.raw`\mathbf{\hat{y}} = a(\cdots a(a(a(\mathbf{X}\mathbf{W}^{<1>T})\mathbf{W}^{<2>T}) \mathbf{W}^{<3>T}) \cdots \mathbf{W}^{<L>T})`}</Latex
  >
  <p>
    This nesting makes it harder to determine the gradients of individual
    weights. Luckily the backpropagation algorithm allows us to calculate the
    gradients of all the weights and biases in a very efficient manner.
  </p>
  <p>
    We can demonstrate how this procedure works using the neural network from
    above.
  </p>
  <p>Let us assume that we receive a batch of inputs containing 3 samples.</p>
  <Latex>
    \mathbf&#123; X &#125; = \begin&#123;bmatrix&#125;
    {#each features as row}
      {#each row as value, idx}
        {value.toFixed(2)}
        {#if idx !== row.length - 1}
          &
        {/if}
      {/each}
      \\
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>
  <p>
    Whe first layer takes a <Latex>3 \times 2</Latex> matrix as input and generates
    a <Latex>3 \times 4</Latex> matrix. For that we utilize the matrix<Latex
      >{String.raw`\mathbf{W}`}</Latex
    > and the vector <Latex>{String.raw`\mathbf{b}`}</Latex>.
  </p>
  <Latex>
    \mathbf&#123; W ^&#123; &lt;1&gt; &#125;&#125; = \begin&#123;bmatrix&#125;
    {#each $weightsStore[0] as row}
      {#each row as value, idx}
        {value.toFixed(2)}
        {#if idx !== row.length - 1}
          &
        {/if}
      {/each}
      \\
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>,
  <Latex>
    {String.raw`\mathbf{b}^{<1>}`} = \begin&#123;bmatrix&#125;
    {#each $biasesStore[0] as value, idx}
      {#each value as v}
        {v.toFixed(2)}
        \\
      {/each}
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>
  <p>
    After multiplying the input features <Latex>{String.raw`\mathbf{X}`}</Latex>
    with the weight matrix, adding the bias vector and applying the sigmoid <Latex
      >{String.raw`\mathbf{A} = a(\mathbf{X}\mathbf{W}^T + \mathbf{b})`}</Latex
    > we end up with the outputs for the first layer.
  </p>
  <Latex>
    {String.raw`\mathbf{A}^{<1>}`} = \begin&#123;bmatrix&#125;
    {#each $activationsStore[0] as row}
      {#each row as value, idx}
        {value.toFixed(2)}
        {#if idx !== row.length - 1}
          &
        {/if}
      {/each}
      \\
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>
  <p>
    We repeat the process taking the outputs from the first layer as input and
    using the weights and biases from the second layer.
  </p>
  <Latex>
    \mathbf&#123; W ^&#123; &lt;2&gt; &#125;&#125; = \begin&#123;bmatrix&#125;
    {#each $weightsStore[1] as row}
      {#each row as value, idx}
        {value.toFixed(2)}
        {#if idx !== row.length - 1}
          &
        {/if}
      {/each}
      \\
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>,
  <Latex>
    {String.raw`\mathbf{b}^{<2>}`} = \begin&#123;bmatrix&#125;
    {#each $biasesStore[1] as value, idx}
      {#each value as v}
        {v.toFixed(2)}
        \\
      {/each}
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>
  <p>
    Finally we use the weights and the bias from the third and final layer to
    attain our predictions.
  </p>
  <Latex>
    \mathbf&#123; W ^&#123; &lt;3&gt; &#125;&#125; = \begin&#123;bmatrix&#125;
    {#each $weightsStore[2] as row}
      {#each row as value, idx}
        {value.toFixed(2)}
        {#if idx !== row.length - 1}
          &
        {/if}
      {/each}
      \\
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>,
  <Latex>
    {String.raw`\mathbf{b}^{<3>}`} = \begin&#123;bmatrix&#125;
    {#each $biasesStore[2] as value, idx}
      {#each value as v}
        {v.toFixed(2)}
        \\
      {/each}
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>
  <p>
    The last activations <Latex>{String.raw`\mathbf{A}^{<3>}`}</Latex> correspond
    to the probabilities to belong to the category 1. Eventually these outputs are
    used in the calculation of the loss function, the cross-entropy.
  </p>
  <Latex>
    {String.raw`\mathbf{\hat{y}} = \mathbf{A}^{<3>}`} = \begin&#123;bmatrix&#125;
    {#each $activationsStore[2] as row}
      {#each row as value, idx}
        {value.toFixed(4)}
        {#if idx !== row.length - 1}
          &
        {/if}
      {/each}
      \\
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>
  <p>
    You might wonder why the outputs are almost identical, even though in the
    previous section those inputs would correspond to different categories. The
    weights have not been adjusted through training yet, but we will tackle that
    problem in the next section.
  </p>
  <div class="separator" />
</Container>
