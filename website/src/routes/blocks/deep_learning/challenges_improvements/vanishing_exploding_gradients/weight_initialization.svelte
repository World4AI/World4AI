<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Weight Initialization</h1>
<div class="separator" />

<Container>
  <p>
    Previously we had mentioned that weights can contribute to vanishing and
    exploding gradients. For the most part we adjust weights in a completely
    automated process by using backpropagation and applying gradient descent.
    For that reason we do not have a lot of influence on how the weights
    develop. The one place where we directly determine the distribution of
    weights is during the initialization process. This section is going to be
    dedicated to weight initialization: the pitfalls and best practices.
  </p>
  <p>
    The first idea we might come up with is to initialize all weights equally,
    specifically to use 0 as the starting value for all weights and biases.
  </p>
  <p>
    We will use this simple neural network to demonstrate the danger of such
    initialization. All we need to do is to work through a single forward and
    backward pass to realize the problem.
  </p>
  <NeuralNetwork />
  <p>
    If we have the same weight <Latex>w</Latex> for all nodes and layers, then in
    the very first forward pass all the neurons from the same layer will produce
    the same value.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
    a_1 &= x_1 * w + x_2 * w \\
    a_2 &= x_1 * w + x_2 * w
  \end{aligned}
    `}</Latex
  >
  <p>
    When we apply backpropagation we will quickly realize that the gradients for
    each of the weights are identical for each node in a particular layer.
  </p>
  <Latex
    >{String.raw`
    \dfrac{\partial L}{\partial o}
    \dfrac{\partial o}{\partial z}
    \dfrac{\partial z}{\partial a_1}
    =
    \dfrac{\partial L}{\partial o}
    \dfrac{\partial o}{\partial z}
    \dfrac{\partial z}{\partial a_2}
    `}</Latex
  >
  <p>
    That means that each layer will always have weights of equal size. This is
    no different than having a neural network with a single neuron per layer.
    The network will never be able to solve complex problems. And if you
    initialize all your weights with zero, the network will always have dead
    neurons, always staying at the 0 value.
  </p>
  <p class="danger">
    Never initialize your weights uniformly. Break the symmetry!
  </p>
  <p>Initialize using the standard normal distribution</p>
  <p>Xavier|Glorot Initialization</p>
  <p>Kaiming|He Initialization</p>
</Container>
