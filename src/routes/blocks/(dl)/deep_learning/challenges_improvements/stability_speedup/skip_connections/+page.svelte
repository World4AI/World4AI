<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Diagram from "$lib/diagram/Diagram.svelte";
  import Latex from "$lib/Latex.svelte";

  const references = [
    {
      author: "K. He, X. Zhang, S. Ren and J. Sun",
      title: "Deep Residual Learning for Image Recognition",
      journal:
        "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      year: "2016",
      pages: "770-778",
      volume: "",
      issue: "",
    },
  ];

  let usualDiagram = [
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 15,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 25 },
        { x: 50, y: 35 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 45,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 55 },
        { x: 50, y: 65 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 75,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 85 },
        { x: 50, y: 95 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 105,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 115 },
        { x: 50, y: 125 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 135,
      width: 55,
      height: 13,
    },
  ];

  let usualDiagramBackward = [
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 15,
      width: 55,
      height: 13,
      color: "var(--main-color-1)",
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 45,
      width: 55,
      height: 13,
      color: "var(--main-color-1)",
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 75,
      width: 55,
      height: 13,
      color: "var(--main-color-1)",
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 105,
      width: 55,
      height: 13,
      color: "var(--main-color-1)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 125 },
        { x: 50, y: 115 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 135,
      width: 55,
      height: 13,
    },
  ];

  let skipDiagram = [
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 10,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 20 },
        { x: 50, y: 30 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 25 },
        { x: 10, y: 25 },
        { x: 10, y: 60 },
        { x: 42, y: 60 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 40,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 48 },
        { x: 50, y: 53 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "plus",
      x: 50,
      y: 60,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 65 },
        { x: 50, y: 70 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 80,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 90 },
        { x: 50, y: 100 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 95 },
        { x: 10, y: 95 },
        { x: 10, y: 130 },
        { x: 42, y: 130 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 110,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 118 },
        { x: 50, y: 123 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "plus",
      x: 50,
      y: 130,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 135 },
        { x: 50, y: 140 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 150,
      width: 55,
      height: 13,
    },
  ];

  let skipDiagramBackward = [
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 10,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 30 },
        { x: 50, y: 20 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "arrow",
      data: [
        { x: 42, y: 60 },
        { x: 10, y: 60 },
        { x: 10, y: 25 },
        { x: 45, y: 25 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 40,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 55 },
        { x: 50, y: 50 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "plus",
      x: 50,
      y: 60,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 73 },
        { x: 50, y: 67 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 80,
      width: 55,
      height: 13,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 100 },
        { x: 50, y: 90 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "arrow",
      data: [
        { x: 42, y: 130 },
        { x: 10, y: 130 },
        { x: 10, y: 95 },
        { x: 45, y: 95 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 110,
      width: 55,
      height: 13,
      color: "var(--main-color-1)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 125 },
        { x: 50, y: 120 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "plus",
      x: 50,
      y: 130,
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 143 },
        { x: 50, y: 137 },
      ],
      moving: true,
      dashed: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 150,
      width: 55,
      height: 13,
    },
  ];

  let additionalLayersDiagram = [
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 15,
      width: 55,
      height: 13,
      color: "var(--main-color-3)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 25 },
        { x: 50, y: 35 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 45,
      width: 55,
      height: 13,
      color: "var(--main-color-3)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 55 },
        { x: 50, y: 65 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 75,
      width: 55,
      height: 13,
      color: "var(--main-color-3)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 85 },
        { x: 50, y: 95 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 105,
      width: 55,
      height: 13,
      color: "var(--main-color-3)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 115 },
        { x: 50, y: 125 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 135,
      width: 55,
      height: 13,
      color: "var(--main-color-3)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 145 },
        { x: 50, y: 155 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Net Input",
      x: 50,
      y: 165,
      width: 55,
      height: 13,
      color: "var(--main-color-2)",
    },
    {
      type: "arrow",
      data: [
        { x: 50, y: 175 },
        { x: 50, y: 185 },
      ],
      dashed: true,
      moving: true,
    },
    {
      type: "block",
      text: "Activation",
      x: 50,
      y: 195,
      width: 55,
      height: 13,
      color: "var(--main-color-2)",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Skip Connections</title>
  <meta
    name="description"
    content="Skip connections allow us to train very deep neural networks. This architecture alleviates the vanishing gradient problem and deals with the degradation problem at the same time."
  />
</svelte:head>

<h1>Skip Connections</h1>
<div class="separator" />

<Container>
  <p>
    We expect deep neural networks to perform better than their shallow
    counterparts. Deeper architecures have more parameters and should be able to
    model more complex relationships. Yet when we increase the number of layers,
    training becomes impractical and performance deteriorates. While the usual
    suspect is the vanishing gradient problem, He et al.<InternalLink
      type="reference"
      id="1"
    /> were primarily motivated by the so called degradation problem, when they first
    released their ResNet (residual network) architecure. We are going to follow
    both strategies: we will discuss how skip connections might reduce the risk of
    vanishing gradients and we will discuss the degradation problem. As with many
    other techniques in deep learning, we know when a certain architecture works
    empirically, but often we do not know exactly why.
  </p>
  <p>
    Usually data flows from one calculation block into the next, from net inputs
    to activations and vice versa.
  </p>
  <Diagram height="150" components={usualDiagram} />
  <p>
    When we add skip connections, we add an additional path for the data to
    flow. Additionally to flowing into the next net input layer, the output of
    an activation is routed directly into one of the future activation layers.
    The streams from the net input and a previous activation are joined through
    a simple summation and the sum is used as input into the following
    activation function.
  </p>
  <Diagram height="160" components={skipDiagram} />
  <p>
    Neural networks with skip connections are legal and should be able to
    produce the same results, as the vanilla fully connected neural network.
  </p>
  <p>
    Usually when we calculate the output of a neuron, we just pass the net input
    input through the activation function.
  </p>
  <Latex>{String.raw`\mathbf{a}^{<l>} = a(\mathbf{z}^{<l>}_{vanilla})`}</Latex>
  <p>
    With skip connections when we calculate the net input <Latex
      >{String.raw`\mathbf{z}`}</Latex
    > we are actually calculating so called residual values.
  </p>
  <Latex
    >{String.raw`\mathbf{a}^{<l>} = a( \mathbf{a}^{<l-1>} + \mathbf{z}^{<l>}_{skip} )`}</Latex
  >
  <p>
    The residuals are basically the differences between the actual net inputs
    and the outputs from the previous layer.
  </p>
  <Latex
    >{String.raw`
    \mathbf{z}^{<l>}_{skip} =  \mathbf{z}^{<l>}_{vanilla} - \mathbf{a}^{<l-1>}
      `}</Latex
  >
  <p>
    So when we are using skip connections we are not asking how the next layer
    should look like based on the current layer, but we are asking what quantity
    we should add to the current layer to generate the next layer. While the
    question is different, the result should theoretically be the same.
  </p>
  <p>
    Now let's imagine we face the usual problem of vanishing gradients. In a
    certain layer the information flow stops, because the gradient gets close to
    zero. Once that happens, all the preceding layers don't get their gradients
    updated and training essentially stops.
  </p>
  <Diagram height="150" components={usualDiagramBackward} />
  <p>
    If we have skip connections on the other hand, information can flow through
    the additional connection. That way we can circumvent the nodes in the
    neural network, that would deteriorate the performance.
  </p>
  <Diagram height="160" components={skipDiagramBackward} />

  <p>
    The authors of the ResNet paper argued, that the vanishing gradient problem
    has been solved by modern activation functions, weight inintialization
    schemes and batch normalization. The degradation problem therefore had to
    have a different origin.
  </p>
  <p>
    Let's discuss the example below to try to understand the problem. If we
    start with the yellow network and add an additional (blue) layer, we would
    expect the performance to be at least as good as that of the smaller
    (yellow) one.
  </p>
  <Diagram height="210" components={additionalLayersDiagram} />
  <p>
    If the yellow network has already achieved the best performance, the
    addional layer should learn the identity function.
  </p>
  <Latex>{String.raw`\mathbf{a}^{<l-1>} = \mathbf{a}^{<l>}`}</Latex>
  <p>
    That statement should apply, no matter how many additional layer we add.
    Performance should not deteriorate, because the last layers can always learn
    to output the input of the previous layer without change. Yet we know that
    shallow neural networks often outperform their deep counterparts.
  </p>
  <p>
    Maybe it is not as easy to learn the identity function as we imagine. The
    neural network has to find the weights that exactly reproduce the input and
    this is not always a trivial task.
  </p>
  <Latex>{String.raw`\mathbf{a}^{<l>} = a(\mathbf{z}^{<l>}_{vanilla})`}</Latex>
  <p>
    Skip connections on the other hand make it easy for the neural network to
    create an idenity functoin. All the network has to do is to set the weights
    and biases to 0.
  </p>
  <Latex
    >{String.raw`\mathbf{a}^{<l>} = a( \mathbf{a}^{<l-1>} + \xcancel{\mathbf{z}^{<l>}_{skip}})`}</Latex
  >
  <p>
    If we use the ReLU activation function, this equality will hold. Two ReLUs
    in a row do not change the outcome.
  </p>
  <p>
    In practice this technique does not only stop the degradation, but improves
    performance and allows researchers to train very deep neural networks.
  </p>
  <div class="separator" />
</Container>
<Footer {references} />
