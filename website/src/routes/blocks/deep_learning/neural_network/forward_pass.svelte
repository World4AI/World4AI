<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import ForwardPass from "./_forward/ForwardPass.svelte";
  import Latex from "$lib/Latex.svelte";

  let inputs = [
    [0.5, 0.5],
    [0.95, 0.5],
    [0.75, 0.5],
  ];
  let weights = [
    [
      [0.2, 0.3],
      [0.5, 0.5],
      [0.5, 0.1],
      [0.2, 0.3],
    ],
    [
      [1, 0.2, 0.1, -1],
      [0.2, -0.1, -0.3, 0.2],
    ],
    [[0.4, 1]],
  ];
  let biases = [[0, 0.5, 0, 0.5], [1, 0.5], [0.4]];

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  let values = [];
  values.push(JSON.parse(JSON.stringify(inputs)));

  weights.forEach((layer, layerIdx) => {
    let dimValues = [];
    for (let dim = 0; dim < inputs.length; dim++) {
      let valuesLayer = [];
      layer.forEach((weights, weightsIdx) => {
        let zValue = 0;
        weights.forEach((weight, weightIdx) => {
          zValue += weight * values[layerIdx][dim][weightIdx];
        });

        zValue += biases[layerIdx][weightsIdx];
        let value = sigmoid(zValue);
        valuesLayer.push(value);
      });
      dimValues.push(valuesLayer);
    }
    values.push(dimValues);
  });
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
    In the forward pass the features in the dataset are processed layer by layer
    , where the outputs from the previous layer are used as inputs into the
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
<ForwardPass />
<div class="separator" />
<Container>
  <p>
    While the interactive example is a good way to demonstrate the functionality
    of the neural network, we need a way to formalize these calculations through
    mathematical notation.
  </p>
  <p>
    In the above example we worked with a single sample, but in practice we have
    a dataset that is used in the training and testing process. As usual the
    features are represented by the matrix <Latex
      >{String.raw`\mathbf{X}`}</Latex
    > with <Latex>n</Latex> samples (number of rows) and <Latex>m</Latex> features
    (number of columns). Let us imagine the features look as follows.
  </p>
  <Latex>
    \mathbf&#123; X &#125; = \begin&#123;bmatrix&#125;
    {#each values[0] as row}
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
    Each neuron in a particular layer has its own set of weights <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and its own bias <Latex>b</Latex>, even though the neurons have the same
    input. For convenience it makes sence to collect the weights and biases in a
    matrix and a vector respectively.
  </p>
  <p>
    The <Latex>{String.raw`\mathbf{W}`}</Latex> is a <Latex>d \times m</Latex> matrix,
    where <Latex>m</Latex> is as usual the number of features and <Latex
      >d</Latex
    > is the number of neurons in the first hidden layer.
  </p>
  <Latex>
    \mathbf&#123; W &#125; = \begin&#123;bmatrix&#125;
    {#each weights[0] as row}
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
    <Latex>{String.raw`\mathbf{b}`}</Latex> is a <Latex>d \times 1</Latex> vector
    of biases from the first hidden layer.
  </p>
  <Latex>
    \mathbf&#123; b &#125; = \begin&#123;bmatrix&#125;
    {#each biases[0] as value, idx}
      {value.toFixed(2)}
      \\
    {/each}
    \end&#123;bmatrix&#125; \\
  </Latex>
  <p>
    We can calculate the net input by multiplying the matrix of features <Latex
      >{String.raw`\mathbf{X}`}</Latex
    > with the transpose of the weights <Latex>{String.raw`\mathbf{W}^T`}</Latex
    > and adding the weights <Latex>{String.raw`\mathbf{b}`}</Latex>. Finally we
    apply an activation function (sigmoid in our case) to all net inputs, to get
    the activation values <Latex
      >{String.raw`\mathbf{A} = \sigma(\mathbf{Z})`}</Latex
    > .
  </p>

  <Latex
    >{String.raw`\mathbf{Z} = \mathbf{X}\mathbf{W}^T + \mathbf{b} \\ 
    `}</Latex
  >
  <Latex
    >{String.raw`\mathbf{A} = \sigma( \mathbf{Z})
      \\
      `}</Latex
  >
  <p>
    The result is an <Latex>n \times d</Latex> matrix. That means, that for each
    of the samples
    <Latex>n</Latex> we get <Latex>d</Latex> neuron outputs.
  </p>
  <Latex>
    {String.raw`\mathbf{A}`} = \begin&#123;bmatrix&#125;
    {#each values[1] as row}
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
    This process is calculated for each of the hidden and the output layers. To
    make the notation clear regarding the layer we are referring to, we use the
    superscript <Latex>{String.raw`{<l>}`}</Latex>. <Latex>l</Latex> indicates the
    index of the layer of the neural network. Therefore the general calculation for
    a layer looks as follows.
  </p>
  <Latex
    >{String.raw`\mathbf{Z}^{<l>} = \mathbf{A}^{<l-1>}\mathbf{W}^{<l>T} + \mathbf{b}^{<l>} \\ 
    `}</Latex
  >

  <Latex
    >{String.raw`\mathbf{A}^{<l>} = \sigma( \mathbf{Z}^{<l>})
      \\
      `}</Latex
  >
  <p>
    In case we are dealing with the very first layer <Latex>l=1</Latex>, we can
    use the equality <Latex>{String.raw`\mathbf{A}^{l-1} = \mathbf{X}`}</Latex>.
  </p>
  <p>
    Alltogether in the above example we deal with 3 layers of neurons (not
    counting the input layer). The last activations <Latex
      >{String.raw`\mathbf{A}^{<3>}`}</Latex
    > therefore correspond to the probabilities to belong to the category 1. This
    output is used in the calculation of the loss function, the cross entropy.
  </p>
  <Latex>
    {String.raw`\mathbf{\hat{y}} = \mathbf{A}^{<3>}`} = \begin&#123;bmatrix&#125;
    {#each values[3] as row}
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
  <p>Overall our forward pass is a nested calculation.</p>
  <Latex
    >{String.raw`\mathbf{\hat{y}} = \sigma( \sigma(\sigma(\mathbf{X} \mathbf{W}^{<1>T}) \mathbf{W}^{<2>T})\mathbf{W}^{<3>T})`}</Latex
  >
  <p>
    This nesting makes it harder to determine which of the weights have to be
    adjusted in which proportion during gradient descent. Luckily the
    backpropagation algorithm allows us to attribute the loss to individual
    weights.
  </p>
  <div class="separator" />
</Container>
