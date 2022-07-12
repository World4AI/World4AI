<script>
  import Container from "$lib/Container.svelte";
  import Plot from "$lib/Plot.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

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

  function reluPrime(z) {
    return z <= 0 ? 0 : 1;
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
  <title>World4AI | Deep Learning | Activation Functions</title>
  <meta
    name="description"
    content="There are many different activation functions out there, but not all are of equal value. We use sigmoid to scale the values between 0 and 1, we use tanh to scale values between -1 and 1 and we use ReLU almost exclusiviely for all hidden units."
  />
</svelte:head>

<h1>Activation Functions</h1>
<div class="separator" />

<Container>
  <p>
    The derivative of the sigmoid activation function can cause the vanishing
    gradient problem when we increase the number of layers. Over the years
    researchers have come up with activation functions that try to address that
    problem. In this section we are going to compare and contrast some of the
    most popular activation functions, while emphasizing when each of the
    activations should be used.
  </p>
  <div class="separator" />

  <h2>Sigmoid and Softmax</h2>
  <p>
    From our previous discussion it might have seemed, that the sigmoid
    activation function (and by extension softmax) is the root cause of the
    vanishing gradient problem and should be avoided at any cost.
  </p>
  <Plot
    pathsData={sigmoidPath}
    config={{
      width: 500,
      height: 250,
      maxWidth: 800,
      minX: -10,
      maxX: 10,
      minY: -0.01,
      maxY: 1.01,
      xLabel: "z",
      yLabel: "f(z)",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      xTicks: [],
      yTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      numTicks: 21,
    }}
  />
  <p>
    While this is somewhat true, the original argumentation that we used when we
    implemented logistic regression still applies. We can use the sigmoid and
    the softmax to turn logits into probabilities. Nowadays we primarily use the
    sigmoid <Latex>{String.raw`\dfrac{1}{1+e^{-z}}`}</Latex> and the softmax <Latex
      >{String.raw`\dfrac{e^{(-z_k)}}{\sum_d e^{(-z_d)}}`}</Latex
    > in the last layer of the neural network, to determine the probability to belong
    to a particular class.
  </p>
  <p class="info">
    Use the sigmoid and the softmax as activations if you need to scale values
    between 0 and 1.
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
    pathsData={tanhPath}
    config={{
      width: 500,
      height: 250,
      maxWidth: 800,
      minX: -10,
      maxX: 10,
      minY: -1.1,
      maxY: 1.1,
      xLabel: "z",
      yLabel: "f(z)",
      xTicks: [],
      yTicks: [1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, -1],
      numTicks: 21,
    }}
  />

  <p>
    For a long time researchers used the tanh activation function instead of the
    sigmoid, because it worked better in practice. Generally tanh exhibits a
    more favourable derivative function. While the sigmoid (red line) can only
    have very low derivatives of up to 0.25, the tanh (blue line) can exhibit a
    derivative of up to 1, thereby reducing the risk of vanishing gradients.
  </p>
  <p>
    <Plot
      pathsData={sigmoidTanhPrimePath}
      config={{
        width: 500,
        height: 250,
        maxWidth: 800,
        minX: -10,
        maxX: 10,
        minY: 0,
        maxY: 1.01,
        xLabel: "z",
        yLabel: "f'(z)",
        pathsColors: ["var(--main-color-1)", "var(--main-color-2)"],
        xTicks: [],
        yTicks: [1, 0.8, 0.6, 0.4, 0.2, 0],
        numTicks: 21,
      }}
    />
  </p>
  <p>
    Over time researchers found better activations functoions that they prefer
    over tanh, but in case you actually desire outputs between -1 and 1, you
    should use the tanh.
  </p>
  <p class="info">
    Use the tanh as your activation function if you need to scale values between
    -1 and 1.
  </p>
  <div class="separator" />

  <h2>ReLU</h2>
  <p>
    The ReLU (rectified linear unit) is at the same time extremely simple and
    extremely powerful. The function returns its input when the input is
    positive and 0 otherwise.
  </p>
  <Latex
    >{String.raw`
    \text{ReLU} = 
        \begin{cases}
        z & \text{if } z > 0 \\
            0 & \text{ otherwise }
        \end{cases}
    `}</Latex
  >
  <Plot
    pathsData={reluPath}
    config={{
      width: 500,
      height: 250,
      maxWidth: 800,
      minX: -10,
      maxX: 10,
      minY: -0.1,
      maxY: 10.1,
      xLabel: "z",
      yLabel: "ReLU",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      pathsColors: ["var(--main-color-1)", "var(--main-color-2)"],
      xTicks: [],
      yTicks: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
      numTicks: 21,
    }}
  />
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
    exactly 0. Given there is even a single neuron in the chain with a negative
    net input, the whole derivative will amount to 0. This is true and is known
    under the name of <Highlight>dying relu problem</Highlight>, but in practice
    you will not encounter the problem too often. Given that you have a large
    amount of neurons in each layer, there should be enough paths to propagate
    the signal.
  </p>
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
  <p>In the example below alpha is exactly 0.1.</p>
  <Plot
    pathsData={leakyReluPath}
    config={{
      width: 500,
      height: 250,
      maxWidth: 800,
      minX: -10,
      maxX: 10,
      minY: -0.1,
      maxY: 10.1,
      xLabel: "z",
      yLabel: "ReLU",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      pathsColors: ["var(--main-color-1)", "var(--main-color-2)"],
      xTicks: [],
      yTicks: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
      numTicks: 21,
    }}
  />
  <p>
    There are many more activation functions out there, expecially those that
    try to improve the original ReLU. For the most part we will use the plain
    vanilla ReLU, because the mentioned improvements generally do not provide
    significant advantages.
  </p>
  <p class="info">
    You should use the ReLU as your main activation function. Deviate only from
    this activation, if you have any specific reason to do so.
  </p>

  <div class="separator" />
</Container>
