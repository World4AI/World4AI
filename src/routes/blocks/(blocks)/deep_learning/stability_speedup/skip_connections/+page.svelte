<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import SvgContainer from "$lib/SvgContainer.svelte";
  // imports for the diagram
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Plus from "$lib/diagram/Plus.svelte";
  import skipMetrics from "./skip_metrics.png";

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
</script>

<svelte:head>
  <title>Skip Connections - World4AI</title>
  <meta
    name="description"
    content="Skip connections allow us to train very deep neural networks. This architecture alleviates the vanishing gradient problem and deals with the degradation problem at the same time."
  />
</svelte:head>

<h1>Skip Connections</h1>
<div class="separator" />

<Container>
  <p>
    If we had to pick just one invention in deep learning, that had allowed us
    to train truely deep nearal networks, it would most likely be <Highlight
      >skip connections</Highlight
    >. This technique is the bread and butter of many modern-day AI researchers
    and practicioners. If you removed skip connections from state of the art
    deep learning architectures, most of them would fall apart. So let's have a
    look at them.
  </p>
  <p>
    Usually we expect deep neural networks to perform better than their shallow
    counterparts. Deeper architecures have more parameters and should be able to
    model more complex relationships. Yet when we increase the number of layers,
    training becomes impractical and performance deteriorates. While the usual
    suspect is the vanishing gradient problem, He et al.<InternalLink
      type="reference"
      id="1"
    /> were primarily motivated by the so called <Highlight
      >degradation problem</Highlight
    >, when they developed the ResNet (residual network) architecure. In this
    section we are going to cover both possibilities: we will discuss how skip
    connections might reduce the risk of vanishing gradients and we will discuss
    the degradation problem. As with many other techniques in deep learning, we
    know that a certain architecture works empirically, but often we do not know
    exactly why.
  </p>
  <p>
    In the arcitectures that we covered so far, data flows from one calculation
    block into the next, from net inputs to activations and vice versa.
  </p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 100 150">
      <Block x={50} y={15} width={55} height={13} text="Activation" />
      <Arrow
        data={[
          { x: 50, y: 25 },
          { x: 50, y: 35 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={45} width={55} height={13} text="Net Input" />
      <Arrow
        data={[
          { x: 50, y: 55 },
          { x: 50, y: 65 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={75} width={55} height={13} text="Activation" />
      <Arrow
        data={[
          { x: 50, y: 85 },
          { x: 50, y: 95 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={105} width={55} height={13} text="Net Input" />
      <Arrow
        data={[
          { x: 50, y: 115 },
          { x: 50, y: 125 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={135} width={55} height={13} text="Activation" />
    </svg>
  </SvgContainer>

  <p>
    When we add skip connections, we add an additional path for the data to
    flow. Additionally to flowing into the next net input layer directly, the
    output of an activation is routed directly into one of the future activation
    layers. The streams from the net input and a previous activation are joined
    through a simple summation and the sum is used as input into the following
    activation function.
  </p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 100 160">
      <Block x={50} y={10} width={55} height={13} text="Activation" />
      <Arrow
        data={[
          { x: 50, y: 20 },
          { x: 50, y: 30 },
        ]}
        dashed={true}
        moving={true}
      />
      <Arrow
        data={[
          { x: 50, y: 25 },
          { x: 10, y: 25 },
          { x: 10, y: 60 },
          { x: 42, y: 60 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={40} width={55} height={13} text="Net Input" />
      <Arrow
        data={[
          { x: 50, y: 48 },
          { x: 50, y: 53 },
        ]}
        dashed={true}
        moving={true}
      />
      <Plus x={50} y={60} />
      <Arrow
        data={[
          { x: 50, y: 65 },
          { x: 50, y: 70 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={80} width={55} height={13} text="Activation" />
      <Arrow
        data={[
          { x: 50, y: 90 },
          { x: 50, y: 100 },
        ]}
        dashed={true}
        moving={true}
      />
      <Arrow
        data={[
          { x: 50, y: 95 },
          { x: 10, y: 95 },
          { x: 10, y: 130 },
          { x: 42, y: 130 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={110} width={55} height={13} text="Net Input" />
      <Arrow
        data={[
          { x: 50, y: 118 },
          { x: 50, y: 123 },
        ]}
        dashed={true}
        moving={true}
      />
      <Plus x={50} y={130} />
      <Arrow
        data={[
          { x: 50, y: 135 },
          { x: 50, y: 140 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={150} width={55} height={13} text="Activation" />
    </svg>
  </SvgContainer>
  <p>
    Usually when we calculate the output of a neuron, we just pass the net input
    through the activation function <Latex>f</Latex>.
  </p>
  <Latex>{String.raw`\mathbf{a}^{<l>} = f(\mathbf{z}^{<l>}_{vanilla})`}</Latex>
  <p>
    With skip connections what we actually calculate are the so called residual
    values.
  </p>
  <Latex
    >{String.raw`\mathbf{a}^{<l>} = f(\mathbf{a}^{<l-1>} + \mathbf{z}^{<l>}_{skip} )`}</Latex
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
    Theoretically skip connections should produce the same results, because we
    are not changing our task completely, we are just reformulating it. Yet the
    reality is different, because training deep neaural networks with skip
    connections is easier.
  </p>
  <p>
    Let's imagine we face the usual problem of vanishing gradients. In a certain
    layer the information flow stops, because the gradient gets close to zero.
    Once that happens, all the preceding layers don't get their gradients
    updated due to the chain rule and training essentially stops.
  </p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 100 150">
      <Block
        x={50}
        y={15}
        width={55}
        height={13}
        text="Activation"
        color="var(--main-color-1)"
      />
      <Block
        x={50}
        y={45}
        width={55}
        height={13}
        text="Net Input"
        color="var(--main-color-1)"
      />
      <Block
        x={50}
        y={75}
        width={55}
        height={13}
        text="Activation"
        color="var(--main-color-1)"
      />
      <Block
        x={50}
        y={105}
        width={55}
        height={13}
        text="Net Input"
        color="var(--main-color-1)"
      />
      <Arrow
        data={[
          { x: 50, y: 125 },
          { x: 50, y: 115 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={135} width={55} height={13} text="Activation" />
    </svg>
  </SvgContainer>
  <p>
    If we have skip connections on the other hand, information can flow through
    the additional connection. That way we can circumvent the dead nodes in the
    neural network and the gradients can keep flowing.
  </p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 100 160">
      <Block x={50} y={10} width={55} height={13} text="Activation" />
      <Arrow
        data={[
          { x: 50, y: 32 },
          { x: 50, y: 22 },
        ]}
        dashed={true}
        moving={true}
      />
      <Arrow
        data={[
          { x: 45, y: 60 },
          { x: 10, y: 60 },
          { x: 10, y: 25 },
          { x: 45, y: 25 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={40} width={55} height={13} text="Net Input" />
      <Arrow
        data={[
          { x: 50, y: 55 },
          { x: 50, y: 50 },
        ]}
        dashed={true}
        moving={true}
      />
      <Plus x={50} y={60} />
      <Arrow
        data={[
          { x: 50, y: 73 },
          { x: 50, y: 67 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={80} width={55} height={13} text="Activation" />
      <Arrow
        data={[
          { x: 50, y: 103 },
          { x: 50, y: 90 },
        ]}
        dashed={true}
        moving={true}
      />
      <Arrow
        data={[
          { x: 45, y: 130 },
          { x: 10, y: 130 },
          { x: 10, y: 95 },
          { x: 45, y: 95 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block
        x={50}
        y={110}
        width={55}
        height={13}
        text="Net Input"
        color="var(--main-color-1)"
      />
      <Arrow
        data={[
          { x: 50, y: 125 },
          { x: 50, y: 120 },
        ]}
        dashed={true}
        moving={true}
      />
      <Plus x={50} y={130} />
      <Arrow
        data={[
          { x: 50, y: 142 },
          { x: 50, y: 137 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block x={50} y={150} width={55} height={13} text="Activation" />
    </svg>
  </SvgContainer>
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
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 100 210">
      <Block
        x={50}
        y={15}
        width={55}
        height={13}
        text="Activation"
        color={"var(--main-color-3)"}
      />
      <Arrow
        data={[
          { x: 50, y: 25 },
          { x: 50, y: 35 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block
        x={50}
        y={45}
        width={55}
        height={13}
        text="Net Input"
        color={"var(--main-color-3)"}
      />
      <Arrow
        data={[
          { x: 50, y: 55 },
          { x: 50, y: 65 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block
        x={50}
        y={75}
        width={55}
        height={13}
        text="Activation"
        color={"var(--main-color-3)"}
      />
      <Arrow
        data={[
          { x: 50, y: 85 },
          { x: 50, y: 95 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block
        x={50}
        y={105}
        width={55}
        height={13}
        text="Net Input"
        color={"var(--main-color-3)"}
      />
      <Arrow
        data={[
          { x: 50, y: 115 },
          { x: 50, y: 125 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block
        x={50}
        y={135}
        width={55}
        height={13}
        text="Activation"
        color={"var(--main-color-3)"}
      />
      <Arrow
        data={[
          { x: 50, y: 145 },
          { x: 50, y: 155 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block
        x={50}
        y={165}
        width={55}
        height={13}
        text="Net Input"
        color={"var(--main-color-2)"}
      />
      <Arrow
        data={[
          { x: 50, y: 175 },
          { x: 50, y: 185 },
        ]}
        dashed={true}
        moving={true}
      />
      <Block
        x={50}
        y={195}
        width={55}
        height={13}
        text="Activation"
        color={"var(--main-color-2)"}
      />
    </svg>
  </SvgContainer>
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
  <Latex
    >{String.raw`\mathbf{a}^{<l>} = f(\mathbf{z}^{<l>}_{vanilla}) = \mathbf{a}^{<l-1>} `}</Latex
  >
  <p>
    Skip connections on the other hand make it easy for the neural network to
    create an idenity functoin. All the network has to do is to set the weights
    and biases to 0.
  </p>
  <Latex
    >{String.raw`\mathbf{a}^{<l>} = f(\mathbf{a}^{<l-1>} + \xcancel{\mathbf{z}^{<l>}_{skip}})`}</Latex
  >
  <p>
    If we use the ReLU activation function, the equality above will hold,
    because two ReLUs in a row do not change the outcome.
  </p>
  <p>
    Impelementing skip connectins in PyTorch is a piece of cake. Below we create
    a new module called <code>ResBlock</code>. The block implements a skip
    connection by adding the input of the module to the output of the activation
    function.
  </p>
  <PythonCode
    code={`class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        
    def forward(self, features):
        output = F.relu(self.linear(features))
        return features + output
`}
  />
  <p>
    We implement our model by stacking 20 of residual blocks and we train the
    model on the MNIST dataset.
  </p>
  <PythonCode
    code={`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(NUM_FEATURES, HIDDEN),
                nn.ReLU(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                ResBlock(),
                nn.Linear(HIDDEN, NUM_LABELS),
            )
    
    def forward(self, features):
        return self.layers(features)
`}
  />
  <p>
    While we can observe some overfitting, we do not have any trouble training
    such a deep neaural network.
  </p>
  <img
    src={skipMetrics}
    alt="Metrics of a deep neural network with skip connections"
  />
</Container>
<Footer {references} />
