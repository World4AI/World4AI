<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";

  import reluMetrics from "./relu_metrics.png";
  import sigmoidMetrics from "./sigmoid_metrics.png";

  let references = [
    {
      author: "Glorot, Xavier and Bordes, Antoine and Bengio, Yoshua",
      title:
        "Deep Sparse Rectifier Neural Networks, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics",
      journal: "",
      year: "2011",
      pages: "315-323",
      volume: "15",
      issue: "",
    },
  ];

  //1. sigmoid
  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function sigmoidPrime(z) {
    return sigmoid(z) * (1 - sigmoid(z));
  }

  // fill the sigmoid function data
  const sigmoidPath = [];
  for (let i = -10; i <= 10; i += 0.1) {
    sigmoidPath.push({ x: i, y: sigmoid(i) });
  }

  //2. tanh
  function tanh(z) {
    return (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
  }

  function tanhPrime(z) {
    return 1 - tanh(z) ** 2;
  }

  // fill the tanh function data
  const tanhPath = [];
  for (let i = -10; i <= 10; i += 0.1) {
    tanhPath.push({ x: i, y: tanh(i) });
  }

  const sigmoidTanhPrimePath = [[], []];
  for (let i = -10; i <= 10; i += 0.1) {
    sigmoidTanhPrimePath[0].push({ x: i, y: sigmoidPrime(i) });
    sigmoidTanhPrimePath[1].push({ x: i, y: tanhPrime(i) });
  }

  //3. relu
  function relu(z) {
    return z <= 0 ? 0 : z;
  }

  // fill the ReLU function data
  const reluPath = [];
  for (let i = -10; i <= 10; i += 0.1) {
    reluPath.push({ x: i, y: relu(i) });
  }

  //4. leaky relu
  //relu
  function leakyRelu(z, alpha = 0.1) {
    return z <= 0 ? alpha * z : z;
  }

  // fill the ReLU function data
  const leakyReluPath = [];
  for (let i = -10; i <= 10; i += 0.1) {
    leakyReluPath.push({ x: i, y: leakyRelu(i) });
  }
</script>

<svelte:head>
  <title>Activation Functions - World4AI</title>
  <meta
    name="description"
    content="There are many different activation functions out there, but many encourage vanishing gradients. The sigmoid activation function is one of the main drivers of the vanishing gradients problem. The tanh is a slighly better option, but can still lead to vanishing gradients. ReLU (and its variants) is better suited for hidden units and is therefore the most popular activation function."
  />
</svelte:head>

<h1>Activation Functions</h1>
<div class="separator" />

<Container>
  <p>
    The sigmoid activation function is one of the main causes of the vanishing
    gradients problem. Because of that researchers have tried to come up with
    activation functions with better properties. In this section we are going to
    compare and contrast some of the most popular activation functions, while
    emphasizing when each of the activations should be used.
  </p>
  <div class="separator" />

  <h2>Sigmoid and Softmax</h2>
  <p>
    From our previous discussion it might have seemed, that the sigmoid
    activation function (and by extension softmax) is the root cause of the
    vanishing gradient problem and should be avoided at all cost.
  </p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[-10, 10]}
    range={[0, 1]}
    padding={{ top: 5, right: 40, bottom: 30, left: 40 }}
  >
    <Path data={sigmoidPath} />
    <XLabel text={"z"} type="latex" />
    <YLabel text={"\\sigma(z)"} type="latex" x={0} />
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
      xOffset={-10}
      yOffset={23}
      fontSize={8}
    />
  </Plot>
  <p>
    While this is somewhat true, the original argumentation that we used when we
    implemented logistic regression still applies. We can use the sigmoid and
    the softmax to turn logits into probabilities. Nowadays we primarily use the
    sigmoid <Latex>{String.raw`\dfrac{1}{1+e^{-z}}`}</Latex> and the softmax <Latex
      >{String.raw`\dfrac{e^{z}}{\sum e^{z}}`}</Latex
    > in the last layer of the neural network, to determine the probability to belong
    to a particular class.
  </p>
  <Alert type="info">
    Use the sigmoid and the softmax as activations if you need to scale values
    between 0 and 1.
  </Alert>
  <p>
    There are generally two ways to implement activation functions. We can use
    PyTorch in a functional way and apply <code>torch.sigmoid(X)</code> in the
    <code>forward()</code>
    function of the model or as we have done so far, we can use the object-oriented
    way and use the <code>torch.nn.Sigmoid()</code> as part of
    <code>nn.Sequential()</code>.
  </p>
  <PythonCode
    code={`# functional way
sigmoid_output = torch.sigmoid(X)
# object-oriented way
sigmoid_layer = torch.nn.Sigmoid()
`}
  />
  <p>
    If we can fit the whole logic of the model into <code>nn.Sequential</code>,
    we will generally do that and use the object-oriented way. Sometimes, when
    the code gets more complicated, this will not possible and we will resort to
    the functional approach. The choice is generally yours.
  </p>
  <div class="separator" />

  <h2>Hyperbolic Tangent</h2>
  <p>
    The tanh activation function <Latex
      >{String.raw`\dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}`}</Latex
    > (also called hypterbolic tangent) is similar in spirit to the sigmoid activation
    function. Looking from a distance you might confuse the two, but there are some
    subtle differences.
  </p>
  <p>
    While both functions saturate when we use very low and very high inputs, the
    sigmoid squishes values between 0 and 1, while the tanh squishes values
    between -1 and 1.
  </p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[-10, 10]}
    range={[-1, 1]}
    padding={{ top: 5, right: 40, bottom: 30, left: 40 }}
  >
    <Path data={tanhPath} />
    <XLabel text={"z"} type="latex" />
    <YLabel text={"\\tanh(z)"} type="latex" x={0} />
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]}
      xOffset={-10}
      yOffset={23}
      fontSize={8}
    />
  </Plot>

  <p>
    For a long time researchers used the tanh activation function instead of the
    sigmoid, because it worked better in practice. Generally tanh exhibits a
    more favourable derivative function. While the sigmoid can only have very
    low derivatives of up to 0.25, the tanh can exhibit a derivative of up to 1,
    thereby reducing the risk of vanishing gradients.
  </p>
  <p>
    <Plot
      width={500}
      height={250}
      maxWidth={800}
      domain={[-10, 10]}
      range={[0, 1]}
      padding={{ top: 5, right: 40, bottom: 30, left: 40 }}
    >
      <Path data={sigmoidTanhPrimePath[0]} strokeDashArray={[2, 4]} />
      <Path data={sigmoidTanhPrimePath[1]} />
      <XLabel text={"z"} type="latex" y={240} />
      <YLabel text={"f(z)'"} type="latex" x={0} />
      <Ticks
        xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
        yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
        xOffset={-10}
        yOffset={23}
        fontSize={8}
      />
    </Plot>
  </p>
  <p>
    Over time researchers found better activations functoions that they prefer
    over tanh, but in case you actually desire outputs between -1 and 1, you
    should use the tanh.
  </p>

  <Alert type="info">
    Use the tanh as your activation function if you need to scale values between
    -1 and 1.
  </Alert>
  <p>
    Once again there are two broad approaches to activation functions: the
    functional and the object-oriented one.
  </p>
  <PythonCode
    code={`tanh_output = torch.tanh(X)
tanh_layer = torch.nn.Tanh()`}
  />
  <div class="separator" />

  <h2>ReLU</h2>
  <p>
    The ReLU (rectified linear unit) is at the same time extremely simple and
    extremely powerful. The function returns the unchanged input <Latex>z</Latex
    > as its output when the input value is positive and 0 otherwise <InternalLink
      type={"reference"}
      id="1"
    />.
  </p>
  <Latex
    >{String.raw`
    \text{ReLU}(z) = 
        \begin{cases}
        z & \text{if } z > 0 \\
            0 & \text{ otherwise }
        \end{cases}
    `}</Latex
  >
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[-10, 10]}
    range={[0, 10]}
    padding={{ top: 15, right: 40, bottom: 30, left: 40 }}
  >
    <Path data={reluPath} />
    <XLabel text={"z"} type="latex" />
    <YLabel text={"relu(z)"} type="latex" x={0} />
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
      xOffset={-10}
      yOffset={23}
      fontSize={8}
    />
  </Plot>
  <p>
    The calculation of the derivative is also extremely straightforward. It is
    exactly 1 when the net input <Latex>z</Latex> is above 1 and 0 otherwise. While
    technically we can not differentiate the function at the knick, in practice this
    works very well.
  </p>
  <Latex
    >{String.raw`
    \text{ReLU Derivative} = 
        \begin{cases}
        1 & \text{if } z > 0 \\
            0 & \text{ otherwise }
        \end{cases}
    `}</Latex
  >
  <p>
    Hopefully you will interject at this point and point out, that while the
    derivative of exactly 1 will help to fight the problem of vanishing
    gradients, a derivative of 0 will push the product in the chain rule to
    exactly 0. This is true and is known as the <Highlight
      >dying relu problem</Highlight
    >, but in practice you will not encounter the problem too often. Given that
    you have a large amount of neurons in each layer, there should be enough
    paths to propagate the signal.
  </p>
  <p>PyTorch offers the two approaches for ReLU as well.</p>
  <PythonCode
    code={`relu_output = torch.relu(X)
relu_layer = torch.nn.ReLU()
`}
  />
  <p>
    Over time researchers tried to come up with improvements for the ReLU
    activation. The leaky ReLU for example does not completely kill off the
    signal, but provides a small slope when the net input is negative.
  </p>
  <Latex
    >{String.raw`
    \text{ReLU} = 
        \begin{cases}
        z & \text{if } z > 0 \\
            \alpha * z & \text{ otherwise }
        \end{cases}
    `}</Latex
  >
  <p>In the example below alpha corresponds to 0.1.</p>
  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[-10, 10]}
    range={[0, 10]}
    padding={{ top: 15, right: 40, bottom: 30, left: 40 }}
  >
    <Path data={leakyReluPath} />
    <XLabel text={"z"} type="latex" />
    <YLabel text={"relu(z)"} type="latex" x={0} />
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
      xOffset={-10}
      yOffset={23}
      fontSize={8}
    />
  </Plot>
  <p>
    When activation functions start to get slighly more exotic, you will often
    not find the in the <code>torch</code> namespace directly, but in the
    <code>torch.nn.functional</code> namespace.
  </p>
  <PythonCode
    code={`lrelu_output = torch.nn.functional.leaky_relu(X, negative_slope=0.01)
lrelu_layer = torch.nn.LeakyReLU(negative_slope=0.01)
`}
  />
  <p>
    There are many more activation functions out there, expecially those that
    try to improve the original ReLU. For the most part we will use the plain
    vanilla ReLU, because the mentioned improvements generally do not provide
    significant advantages.
  </p>
  <Alert type="info">
    You should use the ReLU (or its relatives) as your main activation function.
    Deviate only from this activation, if you have any specific reason to do so.
  </Alert>

  <p>
    Now let's have a peak at the difference in the performance between the
    sigmoid and the ReLU activation functions. Once again we are dealing with
    the MNIST dataset, but this time around we create two models, each with a
    different set of activation functions. Both models are larger, than they
    need to be, in order to demonstrate the vanishing gradient problem.
  </p>
  <PythonCode
    code={`class SigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_FEATURES, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, NUM_LABELS)
        )
        
    def forward(self, features):
        return self.layers(features)
    
class ReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_FEATURES, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, NUM_LABELS)
        )
        
    def forward(self, features):
        return self.layers(features)`}
  />

  <p>
    The sigmoid model starts out very slowly and even after 30 iterations has
    not managed to decrease the training loss significantly. If you train the
    same sigmoid model several times, you will notice, that sometimes the loss
    does not decrease at all. It all depends on the starting weights.
  </p>
  <img src={sigmoidMetrics} alt="Metrics with sigmoid activation" />
  <p>
    The loss of the ReLU model on the other hand decreases significantly, thus
    indicating that the gradients propagate much better with this type of
    activation function.
  </p>
  <img src={reluMetrics} alt="Metrics with relu activation" />
</Container>
<Footer {references} />
