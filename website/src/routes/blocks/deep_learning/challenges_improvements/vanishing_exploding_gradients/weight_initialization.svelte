<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  let references = [
    {
      author: "Glorot, Xavier and Bengio Yoshua",
      title:
        "Understanding the Difficulty of Training Deep Feedforward Neural Networks",
      journal:
        "Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, Journal of Machine Learning Research",
      year: "2010",
      pages: "249-256",
      volume: "9",
      issue: "",
    },
    {
      author: "K. He, X. Zhang, S. Ren and J. Sun",
      title:
        " Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",
      journal: "2015 IEEE International Conference on Computer Vision (ICCV)",
      year: "2015",
      pages: "1026-1024",
      volume: "",
      issue: "",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Weight Initialization</title>
  <meta
    name="description"
    content="Proper weight initialization, like (Xavier/Glorot and Kaiming/He) can decrease the chances of exploding or vanishing gradients."
  />
</svelte:head>

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
  <p>
    For a long time researchers were using either a normal distribution (<Latex
      >\mu = 0
    </Latex> and e.g. <Latex>\sigma = 0.1</Latex>) or a uniform distribution
    (e.g in the range <Latex>{String.raw`-0.5 \text{ to } 0.5`}</Latex>) to
    initialize weights. This might seem reasonable, but Glorot and Bengio<InternalLink
      id={1}
      type="reference"
    /> showed that it is much more preferable to initialize weights in such a way,
    that during the forward pass the variance of input neurons and the variance of
    output neurons stays the same and during the backward pass the gradients keep
    a constant variance from layer to layer. That condition reduces the likelihood
    of vanishing or exploding gradients. The authors proposed to initialize weights
    either using a uniform distribution
    <Latex>{String.raw`\mathcal{U}(-a, a)`}</Latex> where <Latex
      >{String.raw`a = \sqrt{\frac{6}{fan_{in} + fan_{out}}}`}</Latex
    > or the normal distribution <Latex
      >{String.raw`\mathcal{N}(0, std^2)`}</Latex
    >, where
    <Latex>{String.raw` std = \sqrt{\frac{2}{fan_{in} + fan_{out}}}`}</Latex>.
    The words <Latex>{String.raw`fan_{in}`}</Latex> and <Latex
      >{String.raw`fan_{out}`}</Latex
    > stand for the number of neurons that go into the layer that we are initializing
    and the number of neurons that is in the layer we are initializing respectively.
  </p>
  <p>
    While the Xavier/Glorot initialization was studied in conjunction with the
    sigmoind and the tanh activation function, the Kaiming/He initialization was
    designed to work with the ReLU activation<InternalLink
      id="2"
      type="reference"
    />. This is the standard initialization mode used in PyTorch.
  </p>
  <p class="info">
    For the most part you will not spend a lot of time dealing with weight
    initializations. Libraries like PyTorch and Keras have good common sense
    initialization values and allow you to switch between the initialization
    modes relatively easy. If you implement backpropagation on your own don't
    forget to at least break the symmetry.
  </p>
</Container>
<Footer {references} />
