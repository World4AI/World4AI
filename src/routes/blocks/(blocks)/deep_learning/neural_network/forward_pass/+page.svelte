<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  const layers = [
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-gray-300" },
        { value: "x_2", class: "fill-gray-300" },
      ],
    },
    {
      title: "Hidden",
      nodes: [
        { value: "a_1", class: "fill-w4ai-yellow" },
        { value: "a_2", class: "fill-w4ai-yellow" },
        { value: "a_3", class: "fill-w4ai-yellow" },
        { value: "a_4", class: "fill-w4ai-yellow" },
      ],
    },
    {
      title: "Output",
      nodes: [{ value: "o_1", class: "fill-w4ai-blue" }],
    },
    {
      title: "Loss",
      nodes: [{ value: "L", class: "fill-w4ai-red" }],
    },
  ];
</script>

<svelte:head>
  <title>Forward Pass - World4AI</title>
  <meta
    name="description"
    content="In the forward pass the neural network takes features as the input, processes them layer by layer and produces estimates of the outputs."
  />
</svelte:head>

<h1>Forward Pass</h1>
<div class="separator" />
<Container>
  <p>
    As the name suggests, during the <Highlight>forward pass</Highlight> information
    flows unidirectionally from the input layer to the output layer. The features
    in the dataset are processed layer by layer until the intended output is generated
    and we can measure the loss.
  </p>
  <p>
    In our circular data example we take the two features as input, process them
    through the hidden layer and produce the probability to belong to one of
    the two categories as an output. This probability is used to measure the
    cross-entropy loss.
  </p>
  <NeuralNetwork {layers} height={150} padding={{ left: 0, right: 10 }} />
  <p>
    While the example above provides an intuitive introduction into the world of
    neural networks we need a way to formalize these calculations through
    mathematical notation.
  </p>
  <p>
    As we have covered in previous chapters, we can calculate the value of a
    neuron <Latex>a</Latex> in a two step process. In the first step we calculate
    the net input
    <Latex>z</Latex> by multiplying the feature vector <Latex
      >{String.raw`\mathbf{x}`}</Latex
    > with the transpose of the weight vector <Latex
      >{String.raw`\mathbf{w}^T`}</Latex
    > and adding the bias scalar <Latex>b</Latex>. In the second step we apply
    an activation function <Latex>f</Latex> to the net input.
  </p>
  <Alert type="info">
    Given that we have a features vector
    <Latex
      >{String.raw`
            \mathbf{x} =
            \begin{bmatrix}
              x_1 & x_2 & x_3 & \cdots & x_m 
            \end{bmatrix}
    `}</Latex
    > and a weight vector
    <Latex
      >{String.raw`
            \mathbf{w} =
            \begin{bmatrix}
              w_1 & w_2 & w_3 & \cdots & w_m 
            \end{bmatrix}
    `}</Latex
    > we can calculate the output of the neuron <Latex>a</Latex> in a two step procedure:
    <div>
      <Latex
        >{String.raw`
      z = \mathbf{x}\mathbf{w}^T + b \\
      a = f(z)
    `}</Latex
      >
    </div>
  </Alert>
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
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{X} =
      \begin{bmatrix}
      x_1^{(1)} & x_2^{(1)} & x_3^{(1)} & \cdots & x_m^{(1)} \\
      x_1^{(2)} & x_2^{(2)} & x_3^{(2)} & \cdots & x_m^{(2)} \\
      x_1^{(3)} & x_2^{(3)} & x_3^{(3)} & \cdots & x_m^{(3)} \\
      \vdots & \vdots & \vdots & \cdots & \vdots \\
      x_1^{(n)} & x_2^{(n)} & x_3^{(n)} & \cdots & x_m^{(n)} \\
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    Similarly we need to be able to deal with several neurons in a layer. Each
    neuron in a particular layer has the same set of inputs, but has its own set
    of weights <Latex>{String.raw`\mathbf{w}`}</Latex> and its own bias <Latex
      >b</Latex
    >. For convenience it makes sence to collect the weights in a matrix <Latex
      >{String.raw`\mathbf{W}`}</Latex
    > and the biases in the vector <Latex>{String.raw`\mathbf{b}`}</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{W} =
      \begin{bmatrix}
      w_1^{[1]} & w_2^{[1]} & w_3^{[1]} & \cdots & w_m^{[1]} \\
      w_1^{[2]} & w_2^{[2]} & w_3^{[2]} & \cdots & w_m^{[2]} \\
      w_1^{[3]} & w_2^{[3]} & w_3^{[3]} & \cdots & w_m^{[3]} \\
      \vdots & \vdots & \vdots & \cdots & \vdots \\
      w_1^{[d]} & w_2^{[d]} & w_3^{[d]} & \cdots & w_m^{[d]} \\
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    The weight matrix <Latex>{String.raw`\mathbf{W}`}</Latex> is a <Latex
      >d \times m</Latex
    > matrix, where <Latex>m</Latex> is the number of features from the previous
    layer and <Latex>d</Latex> is the number of neurons (hidden features) we want
    to calculate for the next layer.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{b} =
      \begin{bmatrix}
      b^{[1]}  &
      b^{[2]} & 
      b^{[3]} & 
      \dots &
      b^{[d]}  
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    <Latex>{String.raw`\mathbf{b}`}</Latex> is a <Latex>1 \times d</Latex> vector
    of biases.
  </p>
  <p>
    We can calculate the net input matrix <Latex>{String.raw`\mathbf{Z}`}</Latex
    > and the activation matrix <Latex>{String.raw`\mathbf{A}`}</Latex> using the
    exact same operations we used before.
  </p>
  <Alert type="info">
    Given that we have a features matrix
    <Latex>{String.raw`\mathbf{X}`}</Latex> and a weight matrix
    <Latex>{String.raw`\mathbf{W}`}</Latex> we can calculate the activations matrix
    <Latex>{String.raw`\mathbf{A}`}</Latex> in a two step procedure:
    <div>
      <Latex
        >{String.raw`
      \mathbf{Z} = \mathbf{X}\mathbf{W}^T + \mathbf{b} \\
      \mathbf{A} = f(\mathbf{Z})
    `}</Latex
      >
    </div>
  </Alert>
  <p>The result is an <Latex>n \times d</Latex> matrix.</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{A} =
      \begin{bmatrix}
      a_1^{(1)} & a_2^{(1)} & a_3^{(1)} & \cdots & a_d^{(1)} \\
      a_1^{(2)} & a_2^{(2)} & a_3^{(2)} & \cdots & a_d^{(2)} \\
      a_1^{(3)} & a_2^{(3)} & a_3^{(3)} & \cdots & a_d^{(3)} \\
      \vdots & \vdots & \vdots & \cdots & \vdots \\
      a_1^{(n)} & a_2^{(n)} & a_3^{(n)} & \cdots & a_d^{(n)} \\
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    Usually we deal with more that a single layer. We distinguish between layers
    by using the superscript <Latex>{String.raw`l`}</Latex>. The matrix <Latex
      >{String.raw`\mathbf{W}^{<1>}`}</Latex
    > for example contains weights that are multiplied with the input features.
  </p>
  <p>For the first layer we get:</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    \mathbf{Z^{<1>}} = \mathbf{X}\mathbf{W^{<1>T}} + \mathbf{b^{<T>}} \\ 
    \mathbf{A^{<1>}} = f( \mathbf{Z^{<l>}})
      `}</Latex
    >
  </div>
  <p>For all the other layers we get:</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    \mathbf{Z^{<l>}} = \mathbf{A^{<l-1>T}}\mathbf{W}^{<l>T} + \mathbf{b}^{<T>} \\ 
    \mathbf{A^{<l>}} = f(\mathbf{Z^{<l>}})
      `}</Latex
    >
  </div>
  <p>We can represent the same idea as a deeply nested function composition.</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{\hat{y}} = a(\cdots a(a(\mathbf{X}\mathbf{W}^{<1>T})\mathbf{W}^{<2>T})  \cdots \mathbf{W}^{<L>T})`}</Latex
    >
  </div>
  <p>
    We keep iterating over matrix multiplications and activation functions,
    until we reach the final layer <Latex>L</Latex>. This nesting of the forward
    pass makes it confusing how we should go about determining the gradients of
    individual weights. Luckily the backpropagation algorithm, that we will
    study in the next section, allows us to calculate the gradients of all the
    weights and biases in a very efficient manner.
  </p>
  <p>
    We can implement the forward pass relatively easy using Python and NumPy.
  </p>
  <PythonCode
    code="A = X
for W, b in zip(weights, biases):
    Z = A @ W.T + b
    A = sigmoid(Z)"
  />
  <div class="separator" />
</Container>
