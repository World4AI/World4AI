<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Alert from "$lib/Alert.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Legend from "$lib/plt/Legend.svelte";

  // neural network parameters
  export let layers = [
    {
      title: "Input",
      nodes: [{ value: "x", class: "fill-white" }],
    },
    {
      title: "Hidden 1",
      nodes: [{ value: "z", class: "fill-white" }],
    },
    {
      title: "",
      nodes: [{ value: "a", class: "fill-white" }],
    },
    {
      title: "Hidden 2",
      nodes: [{ value: "z", class: "fill-white" }],
    },
    {
      title: "",
      nodes: [{ value: "a", class: "fill-white" }],
    },
    {
      title: "Output",
      nodes: [{ value: "z", class: "fill-white" }],
    },
    {
      title: "",
      nodes: [{ value: "a", class: "fill-white" }],
    },
    {
      title: "Loss",
      nodes: [{ value: "L", class: "fill-white" }],
    },
  ];

  //sigmoid
  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  //sigmoid derivative
  function sidmoidGrad(z) {
    return sigmoid(z) * (1 - sigmoid(z));
  }

  let sigmoidPath = [];
  let derivativePath = [];

  for (let i = -10; i <= 10; i += 0.1) {
    sigmoidPath.push({ x: i, y: sigmoid(i) });
    derivativePath.push({ x: i, y: sidmoidGrad(i) });
  }

  let exponentialShrink = [];
  for (let i = 1; i <= 5; i += 0.1) {
    exponentialShrink.push({ x: i, y: 0.25 ** i });
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Exploding/Vanishing Gradients</title>
  <meta
    name="description"
    content="Exploding and vanishing gradients are two common problems in deep learning. By using the chain rule we constantly multiply values below zero, which leads to vanishing gradients or values above zero, which leads to exploding gradients."
  />
</svelte:head>

<h1>Vanishing and Exploding Gradients</h1>
<div class="separator" />
<Container>
  <p>
    We expect the performance of a neural network to improve when we add more
    layers to its architecture. A deep neural network has more degrees of
    freedom to fit to the data than a shallow neural network and should thereby
    perform much better. In the very least the neural network should be able to
    overfit to the training data and to display decent performance on the
    training dataset. Yet the opposite is the case. When you naively keep adding
    more and more layers to the neural network, the performance will start to
    deterioarate until the network is not able to learn anything at all. This
    has to do with the so called <Highlight>vanishing</Highlight> or <Highlight
      >exploding</Highlight
    > gradients. The vanishing gradient problem especially plagued the machine learning
    community for a long period of time, but by now we have some excellent tools
    to deal with those problems.
  </p>
  <p>
    To focus on the core idea of the problem, we are going to assume that each
    layer has just one neuron with one weight and no bias. While this is an
    unreasonable assumption, the ideas will hold for much more complex neural
    networks.
  </p>
</Container>
<NeuralNetwork {layers} maxWidth={"600px"} />

<Container>
  <p>
    The forward pass is straighforward. We iterate between the calculation of
    the net value <Latex>{String.raw`z^{<l>}`}</Latex> and the activation<Latex
      >{String.raw`a^{<l>}`}</Latex
    > until we are able to calculate the final activation <Latex>a_3</Latex> and
    the loss <Latex>L</Latex>.
  </p>
  <Latex
    >{String.raw`
    \begin{aligned}
    z^{<1>} &= x^{<1>}w^{<1>} \\
    a^{<1>} &= a(z^{<1>}) \\
    z^{<2>} &= a^{<1>}w^{<2>} \\
    a^{<2>} &= a(z^{<2>}) \\
    z^{<3>} &= a^{<2>}w^{<3>} \\
    a^{<3>} &= a(z^{<3>}) \\
    \end{aligned}
      `}</Latex
  >
  <p>
    In the backward pass we calculate the derivative of the loss with respect to
    weights of different layers by using the chain rule over and over again. For
    the first weigth <Latex>{String.raw`w^{<1>}`}</Latex> the calculation of the
    derivative would look as follows.
  </p>
  <Latex
    >{String.raw`\dfrac{d}{dw^{<1>}} Loss = 
    \dfrac{dLoss}{da^{<3>}} 
    \boxed{
    \dfrac{da^{<3>}}{dz^{<3>}} 
    \dfrac{dz^{<3>}}{da^{<2>}} 
    \dfrac{da^{<2>}}{dz^{<2>}} 
    \dfrac{dz^{<2>}}{da^{<1>}} 
    \dfrac{da^{<1>}}{dz^{<1>}} 
    }
    \dfrac{dz^{<1>}}{dw^{<1>}} 
    `}</Latex
  >
  <p>
    If you look at the boxed calculations, you should notice that the same type
    of calculations are repeated over and over again: <Latex
      >{String.raw`\dfrac{da}{dz}`}</Latex
    > and <Latex>{String.raw`\dfrac{dz}{da}`}</Latex>. We would encounter the
    same pattern even if we had to deal with 100 layers. If we can figure out
    the nature of those two derivatives we might understand what the value of
    the overall derivative looks like.
  </p>

  <p>
    So far we have exclusively dealt with the sigmoid activation function <Latex
      >{String.raw`\dfrac{1}{1 + e^{-z}}`}</Latex
    >, therefore the derivative <Latex
      >{String.raw`\dfrac{da^{<l>}}{dz^{<l>}}`}</Latex
    > is <Latex>{String.raw`\sigma(1-\sigma)`}</Latex>. If we draw both
    functions we will notice, that the derivative of the sigmoid approaches 0,
    when the net input gets too large or too small. At its peak the derivative
    is exactly 0.25.
  </p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[-10, 10]}
    range={[0, 1]}
  >
    <Path data={sigmoidPath} />
    <Path data={derivativePath} color="var(--main-color-2)" />
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 0.2, 0.25, 0.4, 0.6, 0.8, 1]}
      xOffset={-18}
      yOffset={15}
    />
    <XLabel text={"z"} type="latex" />
    <YLabel text={"f(x)"} type="latex" x={0} />
    <Legend text={"sigmoid"} coordinates={{ x: -9, y: 0.9 }} />
    <Legend
      text={"sigmoid derivative"}
      coordinates={{ x: -9, y: 0.83 }}
      legendColor="var(--main-color-2)"
    />
  </Plot>
  <p>
    If we assume the best case scenario, we can replace
    <Latex
      >{String.raw`
\dfrac{da^{<l>}}{dz^{<l>}} 
    `}</Latex
    > by 0.25 and we end up with the following calculatoin of the derivative.
  </p>
  <Latex
    >{String.raw`\dfrac{d}{dw^{<1>}} Loss = 
    \dfrac{dLoss}{da^{<3>}} 
    \boxed{
    0.25
    \dfrac{dz^{<3>}}{da^{<2>}} 
    0.25
    \dfrac{dz^{<2>}}{da^{<1>}} 
    0.25
    }
    \dfrac{dz^{<1>}}{dw^{<1>}} 
    `}</Latex
  >
  <p>
    Each additional layer in the neural network forces the derivative to shrink
    by at least 4. With just 5 layers we are dealing with the factor close to <Latex
      >0</Latex
    >,<Latex>{String.raw`0.25^{5}`}</Latex>
  </p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[1, 5]}
    range={[0, 0.3]}
  >
    <Path data={exponentialShrink} />
    <Ticks
      xTicks={[0, 1, 2, 3, 4, 5]}
      yTicks={[0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]}
      xOffset={-18}
      yOffset={15}
    />
    <XLabel text={"Layers"} />
    <YLabel text={"Factor"} />
    <Legend text={"sigmoid"} coordinates={{ x: -9, y: 0.9 }} />
    <Legend
      text={"sigmoid derivative"}
      coordinates={{ x: -9, y: 0.83 }}
      legendColor="var(--main-color-2)"
    />
  </Plot>
  <p>
    Given that the sigmoid derivative <Latex
      >{String.raw`\dfrac{da^{<l>}}{dz^{<l>}}`}</Latex
    > is always between 0.25 and 0, we have to assume, that the overall derivative
    <Latex>{String.raw`\dfrac{d}{dw^{<1>}}Loss`}</Latex> approaches 0 when the number
    of layers starts to grow. Layers that are close to the last layer are still able
    to change their respective weights appropriately, but the farther the layers
    are removed from the loss, the closer the multiplicator gets to 0 and the closer
    the derivative gets to 0. The weights of the first layers remain virtually unchanged
    from their initial values, essentially preventing the neural network from learning.
    That is the vanishing gradient problem.
  </p>
  <p>
    The derivative <Latex>{String.raw`\dfrac{dz^{<l>}}{da^{<l-1>}}`}</Latex> on the
    other hand is just the corresponding weight <Latex
      >{String.raw`w^{<l>}`}</Latex
    >. Assuming for example that <Latex>{String.raw`w^{<2>}`}</Latex> and <Latex
      >{String.raw`w^{<3>}`}</Latex
    > are both 0.95, we would deal with the following gradient.
  </p>
  <Latex
    >{String.raw`\dfrac{d}{dw^{<1>}} Loss = 
    \dfrac{dLoss}{da^{<3>}} 
    \boxed{
    \dfrac{da^{<3>}}{dz^{<3>}} 
    0.95 
    \dfrac{da^{<2>}}{dz^{<2>}} 
    0.95 
    \dfrac{da^{<1>}}{dz^{<1>}} 
    }
    \dfrac{dz^{<1>}}{dw^{<1>}} 
    `}</Latex
  >
  <p>
    Here we can make a similar argument that we did with the derivative of the
    sigmoid. Given that the majority of weights are between -1 and 1 and the
    number of layer grows, we will face vanishing gradients.
  </p>
  <p>
    Obviously unlike with the sigmoid, weights do not have any lower or higher
    bounds. All weights could therefore be in the range <Latex
      >{String.raw`w > 1`}</Latex
    > and <Latex>{String.raw`w < - 1`}</Latex>. If each weight corresponds to
    exactly 2, then the gradient will grow exponentially.
  </p>
  <Latex
    >{String.raw`\dfrac{d}{dw^{<1>}} Loss = 
    \dfrac{dLoss}{da^{<3>}} 
    \boxed{
    \dfrac{da^{<3>}}{dz^{<3>}} 
    2
    \dfrac{da^{<2>}}{dz^{<2>}} 
    2
    \dfrac{da^{<1>}}{dz^{<1>}} 
    }
    \dfrac{dz^{<1>}}{dw^{<1>}} 
    `}</Latex
  >
  <p>
    That could make the gradients in the first layers enormous, leading to the
    so called exploding gradient problem. Gradient descent will most likely
    start to diverge and at some point our program will throw an error, as the
    gradient will overflow.
  </p>
  <Alert type="info">
    Derivatives of activation functions and weights have a significant impact on
    whether we can train a deep neural network successfully or not.
  </Alert>
  <div class="separator" />
</Container>
