<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";
  import BackpropGraph from "$lib/backprop/BackpropGraph.svelte";

  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Rectangle from "$lib/plt/Rectangle.svelte";

  import { Value, MLP } from "$lib/Network.js";
  import circular from "./circular.png";

  // computational graph
  let x1 = new Value(0.9);
  x1._name = "Feature 1";
  let x2 = new Value(0.3);
  x2._name = "Feature 2";

  let w1 = new Value(0.2);
  w1._name = "L1 N1 W1";
  let w2 = new Value(0.5);
  w2._name = "L1 N1 W2";
  let b1 = new Value(1);
  b1._name = "L1 N1 B";

  let w3 = new Value(0.3);
  w3._name = "L1 N2 W1";
  let w4 = new Value(0.7);
  w4._name = "L1 N2 W2";
  let b2 = new Value(0);
  b2._name = "L1 N2 B";

  let mul1 = x1.mul(w1);
  let mul2 = x2.mul(w2);
  let add1 = mul1.add(mul2);
  let add2 = add1.add(b1);
  let neuron1 = add2.sigmoid();
  neuron1._name = "a_1";

  let mul3 = x1.mul(w3);
  let mul4 = x2.mul(w4);
  let add3 = mul3.add(mul4);
  let add4 = add3.add(b2);
  let neuron2 = add4.sigmoid();
  neuron2._name = "a_2";

  let w5 = new Value(0.22);
  w5._name = "L2 N1 W1";
  let w6 = new Value(0.42);
  w6._name = "L2 N1 W2";
  let b3 = new Value(0.2);
  b3._name = "L2 N1 B";

  let mul5 = neuron1.mul(w5);
  let mul6 = neuron2.mul(w6);
  let add5 = mul5.add(mul6);
  let add6 = add5.add(b3);
  let neuron3 = add6.sigmoid();
  neuron3._name = "O";

  neuron3.backward();
  let out = JSON.parse(JSON.stringify(neuron3));
  neuron2._prev = [];
  neuron1._prev = [];

  // neural network that is used to demonstrate autodiff
  const graphLayers = [
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-gray-300" },
        { value: "x_2", class: "fill-gray-300" },
      ],
    },
    {
      title: "Hidden",
      nodes: [
        { value: "a_1", class: "fill-w4ai-yellow" },
        { value: "a_2", class: "fill-w4ai-yellow" },
      ],
    },
    {
      title: "Output",
      nodes: [{ value: "o", class: "fill-w4ai-blue" }],
    },
  ];

  // nn that is actually used to solve the circular problem
  const layers = [
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-gray-300" },
        { value: "x_2", class: "fill-gray-300" },
      ],
    },
    {
      title: "Hidden",
      nodes: [
        { value: "a_1", class: "fill-w4ai-yellow" },
        { value: "a_2", class: "fill-w4ai-yellow" },
        { value: "a_3", class: "fill-w4ai-yellow" },
        { value: "a_4", class: "fill-w4ai-yellow" },
      ],
    },
    {
      title: "Output",
      nodes: [{ value: "o", class: "fill-w4ai-blue" }],
    },
    {
      title: "Loss",
      nodes: [{ value: "L", class: "fill-w4ai-red" }],
    },
  ];

  // create the data to draw the svg
  let pointsData = [[], []];
  let radius = [0.45, 0.25];
  let centerX = 0.5;
  let centerY = 0.5;
  let numPerCategory = 200;

  let Xs = [];
  let ys = [];
  for (let i = 0; i < radius.length; i++) {
    for (let point = 0; point < numPerCategory; point++) {
      //data for drawing
      let angle = 2 * Math.PI * Math.random();
      let r = radius[i];
      let x = r * Math.cos(angle) + centerX;
      let y = r * Math.sin(angle) + centerY;
      pointsData[i].push({ x, y });

      //data for training
      Xs.push([x, y]);
      ys.push(i);
    }
  }

  // determine the x and y coordinates that are going to be used for heatmap
  let numbers = 50;
  let heatmapCoordinates = [];
  for (let i = 0; i < numbers; i++) {
    for (let j = 0; j < numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let coordinate = [];
      coordinate.push(x);
      coordinate.push(y);
      heatmapCoordinates.push(coordinate);
    }
  }

  function shuffle(Xs, ys) {
    for (let i = Xs.length - 1; i > 0; i--) {
      let j = Math.floor(Math.random() * (i + 1));

      [Xs[i], Xs[j]] = [Xs[j], Xs[i]];
      [ys[i], ys[j]] = [ys[j], ys[i]];
    }
  }

  const alpha = 0.8;
  const nin = 2;
  const nouts = [4, 1];
  let lossData = [];
  let heatmapData = [[], []];

  function train() {
    let mlp = new MLP(nin, nouts);
    let loss = new Value(0);
    let epoch = 0;

    function step() {
      epoch += 1;
      //shuffle(Xs, ys);
      for (let i = 0; i < Xs.length; i++) {
        let out = mlp.forward(Xs[i]);
        //cross-entropy
        if (ys[i] === 0) {
          let one = new Value(1);
          loss = loss.add(one.sub(out).log());
        } else if (ys[i] === 1) {
          loss = loss.add(out.log());
        }
      }

      //calculate cross entropy
      loss = loss.neg().div(Xs.length);
      lossData.push({ x: epoch, y: loss.data });
      lossData = lossData;

      //backprop
      loss.backward();
      //gradient descent
      mlp.parameters().forEach((param) => {
        param.data -= alpha * param.grad;
      });
      //clear gradients
      mlp.zeroGrad();
      //reset loss and accuracy
      loss = new Value(0);

      // create heatmap
      let class0 = [];
      let class1 = [];
      heatmapCoordinates.forEach((coordinates) => {
        let pred = mlp.forward(coordinates);
        if (pred.data < 0.5) {
          class0.push({ x: coordinates[0], y: coordinates[1] });
        } else {
          class1.push({ x: coordinates[0], y: coordinates[1] });
        }
      });
      heatmapData = [];
      heatmapData.push(class0);
      heatmapData.push(class1);
    }
    return step;
  }
  let takeStep = train();

  const code1 = `import torch
import numpy as np
import matplotlib.pyplot as plt`;
  const code2 = `# create circular data
def circular_data():
    radii = [0.45, 0.25]
    center_x = 0.5
    center_y = 0.5
    num_points = 200
    X = []
    y = []
    
    for label, radius in enumerate(radii):
        for point in range(num_points):
            angle = 2 * np.pi * np.random.rand()
            feature_1 = radius * np.cos(angle) + center_x
            feature_2 = radius * np.sin(angle) + center_y
            
            X.append([feature_1, feature_2])     
            y.append([label])
            
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)`;
  const code3 = `X, y = circular_data()`;
  const code4 = `feature_1 = X.T[0]
feature_2 = X.T[1]
# plot the circle
plt.figure(figsize=(4,4))
plt.scatter(x=feature_1, y=feature_2, c=y)
plt.title("Circular Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()`;
  const code5 = `class NeuralNetwork:
    
    def __init__(self, X, y, shape=[2, 4, 2, 1], alpha = 0.1):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.weights = []
        self.biases = []
        
        # initialize weights and matrices with random numbers
        for num_features, num_neurons in zip(shape[:-1], shape[1:]):
            weight_matrix = torch.randn(num_neurons, num_features, requires_grad=True)
            self.weights.append(weight_matrix)
            bias_vector = torch.randn(1, num_neurons, requires_grad=True)
            self.biases.append(bias_vector)
        
    def forward(self):
        A = X
        for W, b in zip(self.weights, self.biases):
            Z = A @ W.T + b
            A = torch.sigmoid(Z)
        return A
                    
    def loss(self, y_hat):
        loss =  -(self.y * torch.log(y_hat) + (1 - self.y) * torch.log(1 - y_hat)).mean()
        return loss
                            
    # update weights and biases
    def step(self):
        with torch.inference_mode():
            for w, b in zip(self.weights, self.biases):
                # gradient descent
                w.data.sub_(w.grad * self.alpha)
                b.data.sub_(b.grad * self.alpha)

                # zero out the gradients
                w.grad.zero_()
                b.grad.zero_()`;
  const code6 = `nn = NeuralNetwork(X, y)`;
  const code7 = `# training loop
for i in range(50_000):
    y_hat = nn.forward()
    loss = nn.loss(y_hat)
    if i % 10000 == 0:
        print(loss.data)
    loss.backward()
    nn.step()`;
</script>

<svelte:head>
  <title>Neural Network Training - World4AI</title>
  <meta
    name="description"
    content="We can train a neural network by switching between the forward pass, the backward pass and gradient descent for a number of iterations. The implementation is relatively straightforward if we utilize one of the deep learning packages like PyTorch."
  />
</svelte:head>

<h1>Neural Network Training</h1>
<div class="separator" />

<Container>
  <p>
    Training a neural network is not much different from training logistic
    regression. We have to construct a computational graph first, which will
    allow us to apply the chain rule while propagating the the gradients from
    the loss function all the way to the weights and biases.
  </p>
  <p>
    To emphasise this idea again, we are going to use an example of a neural
    network with two neurons in the hidden layer and a single output neuron. As
    usual we will assume a single training sample to avoid overcomplicated
    computational graphs.
  </p>
  <NeuralNetwork
    layers={graphLayers}
    height={80}
    padding={{ left: 0, right: 10 }}
  />
  <p>
    Let's first zoom into the output neuron of the neural network. We disregard
    the loss function for the moment to keep things simple, but keep in mind,
    that the full graph would contain cross-entropy or the mean squared error.
  </p>
  <p>
    We use the (L)ayer (N)euron (W)eight/(B)ias notation for weights and biases.
    L2 N1 W2 for example stands for weight 2 of the first neuron in the second
    layer of the neural network.
  </p>
  <BackpropGraph graph={neuron3} width={580} height={900} maxWidth={400} />
  <p>
    If you look at the above graph, you should notice, that this neuron is not
    different from a plain vanilla logistic regression graph. Yet instead of
    using the input features to calculate the output, we use the hidden features <Latex
      >a_1</Latex
    > and <Latex>a_2</Latex>. Each of the hidden features is based on a
    different logistic regression with its own set of weights and a bias. So
    when we use backpropagation we do not stop at <Latex>a_1</Latex> or <Latex
      >a_2</Latex
    >, but keep moving towards the earlier weights and biases.
  </p>
  <BackpropGraph graph={out} width={1200} height={1800} maxWidth={700} />
  <p>
    The above graph only includes two hidden sigmoid neurons, but theoretically
    a graph can contains hundreds of layers with hundreds of neurons each.
    Automatic differentiation libraries will automatically construct a
    computational graph and calculate the gradients, no matter the size of the
    neural network.
  </p>
  <p>
    Now let's remember that our original goal is to solve a non linear problem
    of the below kind.
  </p>
  <Plot width={500} height={500} maxWidth={500} domain={[0, 1]} range={[0, 1]}>
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-15}
      yOffset={15}
    />
    <Circle data={pointsData[0]} />
    <Circle data={pointsData[1]} color="var(--main-color-2)" />
    <XLabel text="Feature 1" fontSize={15} />
    <YLabel text="Feature 2" fontSize={15} />
  </Plot>
  <p>
    Our neural network will take the two features as input, process them through
    the hidden layer with four neurons and finally produce the output neuron,
    which contains the probability to belong to one of the two categories. This
    probability is used to measure the cross-entropy loss.
  </p>
  <NeuralNetwork {layers} height={150} padding={{ left: 0, right: 10 }} />
  <p>
    In the example below you can observe how the decision boundary moves when
    you use backpropagation. Usually 10000 steps are sufficient to find weights
    for a good decision boundary, this might take a couple of minutes. Try to
    observe how the cross-entropy and the shape of the decision boundary change
    over time. At a certain point you will most likely see a sharp drop in cross
    entropy, this is when things will start to improve significantly.
  </p>
  <ButtonContainer>
    <PlayButton f={takeStep} delta={0} />
  </ButtonContainer>
  <div class="flex flex-col md:flex-row">
    <Plot
      width={500}
      height={500}
      maxWidth={600}
      domain={[0, 1]}
      range={[0, 1]}
    >
      <Ticks
        xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        xOffset={-15}
        yOffset={15}
      />
      <Rectangle data={heatmapData[0]} size={9} color="var(--main-color-3)" />
      <Rectangle data={heatmapData[1]} size={9} color="var(--main-color-4)" />
      <Circle data={pointsData[0]} />
      <Circle data={pointsData[1]} color="var(--main-color-2)" />
      <XLabel text="Feature 1" fontSize={15} />
      <YLabel text="Feature 2" fontSize={15} />
    </Plot>
    <Plot
      width={500}
      height={500}
      maxWidth={600}
      domain={[0, 12000]}
      range={[0, 1]}
    >
      <Ticks
        xTicks={[0, 2000, 4000, 6000, 8000, 10000, 12000]}
        yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        xOffset={-15}
        yOffset={15}
      />
      <Path data={lossData} />
      <XLabel text="Number of Steps" fontSize={15} />
      <YLabel text="Cross-Entropy Loss" fontSize={15} />
    </Plot>
  </div>

  <div class="separator" />
  <p>
    While the example above provides an intuitive introduction into the world of
    neural networks we need a way to formalize these calculations through
    mathematical notation.
  </p>
  <p>
    As we have covered in previous chapters can calculate the value of a neuron <Latex
      >a</Latex
    > in a two step process. In the first step we calculate the net input
    <Latex>z</Latex> by multiplying the feature vector <Latex
      >{String.raw`\mathbf{x}`}</Latex
    > with the transpose of the weight vector <Latex
      >{String.raw`\mathbf{w}^T`}</Latex
    > and adding the bias scalar <Latex>b</Latex>. In the second step we apply
    an activation function <Latex>f</Latex> to the net input.
  </p>
  <Alert type="info">
    Given that we have a features vector
    <Latex
      >{String.raw`
            \mathbf{x} =
            \begin{bmatrix}
              x_1 & x_2 & x_3 & \cdots & x_m 
            \end{bmatrix}
    `}</Latex
    > and a weight vector
    <Latex
      >{String.raw`
            \mathbf{w} =
            \begin{bmatrix}
              w_1 & w_2 & w_3 & \cdots & w_m 
            \end{bmatrix}
    `}</Latex
    > we can calculate the output of the neuron <Latex>a</Latex> in a two step procedure:
    <div>
      <Latex
        >{String.raw`
      z = \mathbf{x}\mathbf{w}^T + b \\
      a = f(z)
    `}</Latex
      >
    </div>
  </Alert>
  <p>
    In practice we utilize a dataset <Latex>{String.raw`\mathbf{X}`}</Latex> consisting
    of many samples. . As usual <Latex>{String.raw`\mathbf{X}`}</Latex> is an <Latex
      >n \times m</Latex
    > matrix, where <Latex>n</Latex> (rows) is the number of samples and <Latex
      >m</Latex
    > (columns) is the number of input features.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{X} =
      \begin{bmatrix}
      x_1^{(1)} & x_2^{(1)} & x_3^{(1)} & \cdots & x_m^{(1)} \\
      x_1^{(2)} & x_2^{(2)} & x_3^{(2)} & \cdots & x_m^{(2)} \\
      x_1^{(3)} & x_2^{(3)} & x_3^{(3)} & \cdots & x_m^{(3)} \\
      \vdots & \vdots & \vdots & \cdots & \vdots \\
      x_1^{(n)} & x_2^{(n)} & x_3^{(n)} & \cdots & x_m^{(n)} \\
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    Similarly we need to be able to calculate several neurons in a layer. Each
    neuron in a particular layer has the same set of inputs, but has its own set
    of weights <Latex>{String.raw`\mathbf{w}`}</Latex> and its own bias <Latex
      >b</Latex
    >. For convenience it makes sence to collect the weights in a matrix <Latex
      >{String.raw`\mathbf{W}`}</Latex
    > and the biases in the vector <Latex>{String.raw`\mathbf{b}`}</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{W} =
      \begin{bmatrix}
      w_1^{[1]} & w_2^{[1]} & w_3^{[1]} & \cdots & w_m^{[1]} \\
      w_1^{[2]} & w_2^{[2]} & w_3^{[2]} & \cdots & w_m^{[2]} \\
      w_1^{[3]} & w_2^{[3]} & w_3^{[3]} & \cdots & w_m^{[3]} \\
      \vdots & \vdots & \vdots & \cdots & \vdots \\
      w_1^{[d]} & w_2^{[d]} & w_3^{[d]} & \cdots & w_m^{[d]} \\
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    The weight matrix <Latex>{String.raw`\mathbf{W}`}</Latex> is a <Latex
      >d \times m</Latex
    > matrix, where <Latex>m</Latex> is the number of features from the previous
    layer and <Latex>d</Latex> is the number of neurons (hidden features) we want
    to calculate for the next layer.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{b} =
      \begin{bmatrix}
      b^{[1]}  &
      b^{[2]} & 
      b^{[3]} & 
      \dots &
      b^{[d]}  
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    <Latex>{String.raw`\mathbf{b}`}</Latex> is a <Latex>1 \times d</Latex> vector
    of biases.
  </p>
  <p>
    We can calculate the net input matrix <Latex>{String.raw`\mathbf{Z}`}</Latex
    > and the activation matrix <Latex>{String.raw`\mathbf{A}`}</Latex> using the
    exact same operations we used before.
  </p>
  <Alert type="info">
    Given that we have a features matrix
    <Latex>{String.raw`\mathbf{X}`}</Latex> and a weight matrix
    <Latex>{String.raw`\mathbf{W}`}</Latex> we can calculate the activations matrix
    <Latex>{String.raw`\mathbf{A}`}</Latex> in a two step procedure:
    <div>
      <Latex
        >{String.raw`
      \mathbf{Z} = \mathbf{X}\mathbf{W}^T + \mathbf{b} \\
      \mathbf{A} = f(\mathbf{Z})
    `}</Latex
      >
    </div>
  </Alert>
  <p>The result is an <Latex>n \times d</Latex> matrix.</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{A} =
      \begin{bmatrix}
      a_1^{(1)} & a_2^{(1)} & a_3^{(1)} & \cdots & a_d^{(1)} \\
      a_1^{(2)} & a_2^{(2)} & a_3^{(2)} & \cdots & a_d^{(2)} \\
      a_1^{(3)} & a_2^{(3)} & a_3^{(3)} & \cdots & a_d^{(3)} \\
      \vdots & \vdots & \vdots & \cdots & \vdots \\
      a_1^{(n)} & a_2^{(n)} & a_3^{(n)} & \cdots & a_d^{(n)} \\
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    Usually we deal with more that a single layer. We distinguish between layers
    by using the superscript <Latex>{String.raw`l`}</Latex>. The matrix <Latex
      >{String.raw`\mathbf{W}^{<1>}`}</Latex
    > for example contains weights that are multiplied with the input features.
  </p>
  <p>For the first layer we get:</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    \mathbf{Z^{<1>}} = \mathbf{X}\mathbf{W^{<1>T}} + \mathbf{b} \\ 
    \mathbf{A^{<1>}} = f( \mathbf{Z^{<l>}})
      `}</Latex
    >
  </div>
  <p>For all the other layers we get:</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    \mathbf{Z^{<l>}} = \mathbf{A^{<l-1>}}\mathbf{W}^{<l>T} + \mathbf{b} \\ 
    \mathbf{A^{<l>}} = f(\mathbf{Z^{<l>}})
      `}</Latex
    >
  </div>
  <p>We can represent the same idea as a deeply nested function composition.</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{\hat{y}} = f(\cdots f(f(\mathbf{X}\mathbf{W}^{<1>T})\mathbf{W}^{<2>T})  \cdots \mathbf{W}^{<L>T})`}</Latex
    >
  </div>
  <p>
    We keep iterating over matrix multiplications and activation functions,
    until we reach the output layer <Latex>L</Latex>, that is used as input into
    a loss function.
  </p>
  <p>We can implement a neural network relatively easy using PyTorch.</p>
  <PythonCode code={code1} />
  <p>We first create a circular dataset and plot the results.</p>
  <PythonCode code={code2} />
  <PythonCode code={code3} />
  <PythonCode code={code4} />
  <img src={circular} alt="circular data" />
  <p>
    We implement the logic of the neural network by creating a <code
      >NeuralNetwork</code
    > object. WE assume a network with two input neurons, two hidden layers with
    4 and 2 neurons respectively and an output neuron.
  </p>
  <PythonCode code={code5} />
  <p>
    The code is relatively self explanatory. The <code>forward()</code> method
    multiplies the weight matrix of a layer with the features matrix from the
    previous layer and add the bias vector. The <code>loss()</code> method
    calculates the binary cross-entropy and the <code>step()</code> method applies
    gradient descent and zeroes out the gradients.
  </p>
  <PythonCode code={code6} />
  <p>
    Finally we run the forward pass, the backward pass and the gradient descent
    steps in a loop of 50,000 iterations.
  </p>
  <PythonCode code={code7} />
  <pre class="text-sm">
tensor(0.8799)
tensor(0.6817)
tensor(0.0222)
tensor(0.0038)
tensor(0.0019)
  </pre>
  <p>
    The cross-entropy loss reduces drastically and unlike our custom
    implementation above, the PyTorch implementation runs only for a couple of
    seconds.
  </p>
  <div class="separator" />
</Container>
