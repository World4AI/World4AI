<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Plot from "$lib/Plot.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

  //sigmoid
  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  //sigmoid derivative
  function sidmoidGrad(z) {
    return sigmoid(z) * (1 - sigmoid(z));
  }

  let sigmoidPaths = [[], []];

  for (let i = -10; i <= 10; i += 0.1) {
    sigmoidPaths[0].push({ x: i, y: sigmoid(i) });
    sigmoidPaths[1].push({ x: i, y: sidmoidGrad(i) });
  }

  // neural network with 1 neuron
  let width = 650;
  let height = 200;
  let nodeSize = 55;

  const graph = [
    [{ value: "x^{<1>}" }, { value: "w^{<1>}" }],
    [{ value: "z^{<1>}" }],
    [{ value: "a^{<1>}" }, { value: "w^{<2>}" }],
    [{ value: "z^{<2>}" }],
    [{ value: "a^{<2>}" }, { value: "w^{<3>}" }],
    [{ value: "z^{<3>}" }],
    [{ value: "a^{<3>}" }],
    [{ value: "Loss" }],
  ];

  let xGap = (width - nodeSize - 2) / (graph.length - 1);

  // exponential decline

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
    perform much better. Yet the opposite is the case. When you naively keep
    adding more and more layers the neural network, the performance will start
    to deterioarate until the network is not able to learn anything at all. This
    has to do with so called <Highlight>vanishing gradients</Highlight> or <Highlight
      >exploding gradients</Highlight
    >. The vanishing gradient problem plagued the machine learning community for
    a long period of time, but by now we have some excellent tools to deal with
    those problems.
  </p>
  <p>
    To focus on the core idea of the problem, we are going to assume that each
    layer has just one neuron with one weight and no bias. While this is an
    unreasonable assumption, the ideas will hold for much more complex neural
    networks.
  </p>
</Container>

<SvgContainer maxWidth={"700px"}>
  <svg viewBox="0 0 {width} {height}">
    <!-- connections -->
    {#each graph as nodes, nodesIdx}
      {#each nodes as node, nodeIdx}
        {#if nodesIdx !== graph.length - 1}
          {#each graph[nodesIdx + 1] as nextNode, nextNodeIdx}
            {#if nextNodeIdx !== 1}
              <line
                x1={1 + nodesIdx * xGap + nodeSize}
                y1={(height / (nodes.length + 1)) * (nodeIdx + 1)}
                x2={1 + (nodesIdx + 1) * xGap}
                y2={(height / (graph[nodesIdx + 1].length + 1)) *
                  (nextNodeIdx + 1) -
                  nodeSize / 2 +
                  nodeSize / 2}
                stroke="black"
              />
            {/if}
          {/each}
        {/if}
      {/each}
    {/each}
    <!-- nodes -->
    {#each graph as nodes, nodesIdx}
      {#each nodes as node, nodeIdx}
        <rect
          x={1 + nodesIdx * xGap}
          y={(height / (nodes.length + 1)) * (nodeIdx + 1) - nodeSize / 2}
          width={nodeSize}
          height={nodeSize}
          fill="var(--main-color-4)"
          stroke="black"
        />
        <foreignObject
          x={1 + nodesIdx * xGap + nodeSize / 2 - 21}
          y={(height / (nodes.length + 1)) * (nodeIdx + 1) - nodeSize / 2 + 10}
          width={nodeSize}
          height={nodeSize}
        >
          <Latex>{node.value}</Latex>
        </foreignObject>
      {/each}
    {/each}
  </svg>
</SvgContainer>

<Container>
  <p>
    The forward pass is straighforward. We iterate between the calculation of
    the net value <Latex>{String.raw`z^{<l>}`}</Latex> and the activation<Latex
      >{String.raw`a^{<l>}`}</Latex
    > until we are able to calculate the final activation and the loss.
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
    Afterwards we can calculate the derivative of the loss with respect to the
    inintial weight <Latex>{String.raw`w^{<1>}`}</Latex> by using the chain rule
    over and over again.
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
    So far we are exclusively dealing with the sigmoid activatoin function,
    therefore our derivative <Latex
      >{String.raw`\dfrac{da^{<l>}}{dz^{<l>}}`}</Latex
    > is <Latex>{String.raw`\sigma(1-\sigma)`}</Latex>. If we draw both
    functions, the sigmoid as the red line and the derivative of the sigmoid as
    the blue line, we will notice, that the derivative of the sigmoid approaches
    0, when the net input gets too large or too small. In the best case
    scenario, the derivative is exactly 0.25.
  </p>
  <Plot
    pathsData={sigmoidPaths}
    config={{
      width: 500,
      height: 250,
      maxWidth: 800,
      minX: -10,
      maxX: 10,
      minY: 0,
      maxY: 1,
      xLabel: "x",
      yLabel: "f(x)",
      radius: 5,
      pathsColors: ["var(--main-color-1)", "var(--main-color-2)"],
      numTicks: 11,
    }}
  />
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
    by at least 4. With just 5 layers we are dealing with the factor close to 0.
  </p>
  <Plot
    pathsData={exponentialShrink}
    config={{
      width: 500,
      height: 250,
      maxWidth: 800,
      minX: 1,
      maxX: 5,
      minY: 0,
      maxY: 0.3,
      xLabel: "Number of Layers",
      yLabel: "Gradient Multiplicator",
      xTicks: [0, 1, 2, 3, 4, 5],
      yTicks: [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.001],
    }}
  />
  <p>
    Given that the derivative <Latex
      >{String.raw`\dfrac{da^{<l>}}{dz^{<l>}}`}</Latex
    > is going to be between 0.25 and 0, we have to assume, that the overall derivative
    <Latex>{String.raw`\dfrac{d}{dw^{<1>}}Loss`}</Latex> approaches 0 when the number
    of layers starts to grow. Layers that are close to the last layer are still able
    to change their respective weights appropriately, but the farther the layers
    are removed from the loss, the closer the multiplicator gets to 0 and the closer
    the derivative gets to 0. The weights of the first layers remain virtually unchanged
    from their initial values, essentially preventing the neural network from learning.
    That problem is called the vanishing gradient problem.
  </p>
  <p>
    The derivative <Latex>{String.raw`\dfrac{dz^{<l>}}{da^{<l-1>}}`}</Latex> on the
    other hand is just the corresponding weight <Latex
      >{String.raw`w^{<l>}`}</Latex
    >. The gradient of the loss with respect to the weighs depends on all the
    weights that come afterwards. Assuming for example that <Latex
      >{String.raw`w^{<2>}`}</Latex
    > and <Latex>{String.raw`w^{<3>}`}</Latex> are both 0.95, we would deal with
    the following gradient.
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
    exactly 2, then the gradient will grow exponentially, <Latex
      >{String.raw`2^L`}</Latex
    >.
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
  <p class="info">
    Derivatives of activation functions and weights have a significant impact on
    whether we can train a deep neural network successfully or not. Adjusting
    those appropriately is key to success.
  </p>
  <div class="separator" />
</Container>
