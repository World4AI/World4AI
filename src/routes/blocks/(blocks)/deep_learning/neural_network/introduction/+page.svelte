<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Latex from "$lib/Latex.svelte";

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
  ];
  const neuronLayers = [
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
    hierarchy of layers and combines them into a single model. The outputs of a
    layer are used as inputs into the next layer.
  </Alert>
  <p>
    While there are many different activation functions out there, we will
    assume for now that we are dealing with the sigmoid activation function.
    That means, that a neuron is essentially a separate logistic regression unit
    with individual weights and a bias. The neuron below for example takes
    features <Latex>x_1 - x_4</Latex> as inputs, multiplies those with individual
    weights <Latex>w_1 - w_4</Latex>, adds the bias <Latex>b</Latex> and applies
    the sigmoid activation function
    <Latex>\sigma</Latex>.
  </p>
  <NeuralNetwork
    layers={neuronLayers}
    height={130}
    padding={{ left: 0, right: 50 }}
  />
  <p>
    We combine many of such neurons to a whole network of neurons. In the
    example below the neural network calculates 3 neurons in the hidden layer.
    The neurons take the same features as input, but use different weights and
    biases. The outputs <Latex>a_1 - a_3</Latex> are used as inputs into the final
    logistic regression neuron, to produce the probability estimation <Latex
      >o_1</Latex
    >.
  </p>
  <NeuralNetwork
    layers={nnLayers}
    height={130}
    padding={{ left: 0, right: 10 }}
  />

  <p>
    We can train neural networks that can classify images, generate text or play
    computer games. No matter what task we are trying to accomplish and how the
    neural network is structured, the training process of neural networks is
    always done using the same procedure consisting of two major steps.
  </p>
  <p>
    In the <Highlight>forward pass</Highlight>
    the features are processed layer by layer and neuron by neuron to finally determine
    the loss of the neural network.
  </p>
  <NeuralNetwork {layers} height={150} />
  <p>
    In the <Highlight>backward pass</Highlight> the loss is distributed among all
    neurons and the weights and biases are adjusted proportional to their contribution
    to the loss.
  </p>
  <NeuralNetwork
    {layers}
    height={150}
    speed={0.5}
    connectionStyle={"stroke-red-500"}
  />
  <p>
    The forward and the backward pass run back-to-back, until some condition is
    met and the training process is terminated. No matter how complicated the
    neural network looks like on the surface, the forward and the backward pass
    are always present in modern day deep learning.
  </p>
  <div class="separator" />
</Container>
