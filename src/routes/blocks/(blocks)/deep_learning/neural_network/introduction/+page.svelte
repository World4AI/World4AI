<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Latex from "$lib/Latex.svelte";

  const singleNeuron = [
    {
      title: "Features",
      nodes: [
        { value: "x_1", class: "fill-gray-200" },
        { value: "x_2", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
      ],
    },
    {
      title: "Neuron",
      nodes: [{ value: "a", class: "fill-blue-400" }],
    },
  ];

  const manyNeurons = [
    {
      title: "Features",
      nodes: [
        { value: "x_1", class: "fill-gray-200" },
        { value: "x_2", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
      ],
    },
    {
      title: "Neurons Layer",
      nodes: [
        { value: "a_1", class: "fill-blue-400" },
        { value: "a_2", class: "fill-blue-400" },
        { value: "a_3", class: "fill-blue-400" },
      ],
    },
  ];

  const nnLayers = [
    {
      title: "Features",
      nodes: [
        { value: "x_1", class: "fill-gray-200" },
        { value: "x_2", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
      ],
    },
    {
      title: "Hidden Layer 1",
      nodes: [
        { value: "a_1^1", class: "fill-blue-400" },
        { value: "a_2^1", class: "fill-blue-400" },
        { value: "a_3^1", class: "fill-blue-400" },
      ],
    },
    {
      title: "Hidden Layer 2",
      nodes: [
        { value: "a_1^2", class: "fill-blue-400" },
        { value: "a_2^2", class: "fill-blue-400" },
      ],
    },
    {
      title: "Output Layer",
      nodes: [{ value: "o_1", class: "fill-yellow-400" }],
    },
  ];
  const layers = [
    {
      title: "Features",
      nodes: [
        { value: "x_1", class: "fill-gray-200" },
        { value: "x_2", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
        { value: "x_3", class: "fill-gray-200" },
      ],
    },
    {
      title: "Hidden Layer",
      nodes: [
        { value: "a_1", class: "fill-blue-400" },
        { value: "a_2", class: "fill-blue-400" },
        { value: "a_3", class: "fill-blue-400" },
      ],
    },
    {
      title: "Output Layer",
      nodes: [{ value: "o_1", class: "fill-yellow-400" }],
    },
    {
      title: "Loss",
      nodes: [{ value: "L", class: "fill-red-500" }],
    },
  ];
</script>

<svelte:head>
  <title>Neural Network Introduction - World4AI</title>
  <meta
    name="description"
    content="A neural network is an object, that combines layers of neurons into a single model. The training process of a neural network consists of a forward and a backward pass. In the forward pass the features are used to calculate the loss of the function. In the backward pass the loss is distributed among individual neurons."
  />
</svelte:head>

<h1>Neural Network</h1>
<div class="separator" />
<Container>
  <p>
    Let's start this chapter with the obvious question: <Highlight
      >what is a neural network?</Highlight
    >
  </p>
  <Alert type="info"
    >A neural network is an object, that structures individual neurons in a
    hierarchy of layers and combines them into a single model, by feeding the
    outputs of a layer as inputs into the next layer.
  </Alert>
  <p>
    If the above definition does not make any sense to you, below is a more
    intuitive explanation.
  </p>
  <p>
    There are many different activation functions out there, but for now we will
    assume that we are dealing with the sigmoid activation function. That means,
    that a neuron is essentially a separate logistic regression unit with
    individual weights and a bias. The neuron below for example takes features <Latex
      >x_1 - x_4</Latex
    > as inputs, multiplies those with individual weights <Latex
      >w_1 - w_4</Latex
    >, adds the bias <Latex>b</Latex> and applies the sigmoid activation function
    <Latex>\sigma</Latex>.
  </p>
  <NeuralNetwork
    layers={singleNeuron}
    height={130}
    padding={{ left: 0, right: 50 }}
  />
  <p>
    In a neural network the same are used to produce several different neurons.
    Those neurons utilize different weights and biases and produce therefore
    different outputs. Such a collection of neurons is called a <Highlight
      >layer</Highlight
    >.
  </p>
  <NeuralNetwork
    layers={manyNeurons}
    height={130}
    padding={{ left: 0, right: 50 }}
  />
  <p>
    We can stack several layers after each other to produce a neural network.
    The outputs of the previous layer are used as inputs instead of the input
    features. Often the input neurons are also called <Highlight
      >hidden features</Highlight
    >.
  </p>
  <NeuralNetwork
    layers={nnLayers}
    height={130}
    padding={{ left: 0, right: 10 }}
  />
  <p>
    The output neuron(s) is (are) used as an input into the loss function, for
    example the cross-entropy loss if we are dealing with a classificatio
    problem.
  </p>

  <p>
    We can train neural networks that can classify images, generate text or play
    computer games. No matter what task we are trying to accomplish and how the
    neural network is structured, the training process of neural networks is
    always done using the same steps that we used in linear and logistic
    regression.
  </p>
  <p>
    In the <Highlight>forward pass</Highlight>
    the features are processed layer by layer and neuron by neuron to finally determine
    the loss of the neural network and to construct a computational graph.
  </p>
  <NeuralNetwork {layers} height={150} />
  <p>
    In the <Highlight>backward pass</Highlight> we use the backpropagation algorithm
    to calculate the gradients for all weights and biases.
  </p>
  <NeuralNetwork
    {layers}
    height={150}
    speed={0.5}
    connectionStyle={"stroke-red-500"}
  />
  <p>
    Conceptually the whole learning process is not much different from what we
    saw in the previous chapters. The computational graph is larger and broader,
    but the ideas are the same.
  </p>
  <div class="separator" />
</Container>
