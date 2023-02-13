<script>
  import Container from "$lib/Container.svelte";
  import Alert from "$lib/Alert.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import BackpropGraph from "$lib/backprop/BackpropGraph.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import { Value } from "$lib/Network.js";

  import xavierMetrics from "./xavier_metrics.png";

  const layers = [
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-gray-300" },
        { value: "x_2", class: "fill-gray-300" },
      ],
    },
    {
      title: "Hidden Layer",
      nodes: [
        { value: "a_1", class: "fill-gray-300" },
        { value: "a_2", class: "fill-gray-300" },
      ],
    },
    {
      title: "Output",
      nodes: [{ value: "o", class: "fill-gray-300" }],
    },
  ];

  const layers2 = [
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-gray-300" },
        { value: "x_2", class: "fill-gray-300" },
      ],
    },
    {
      title: "Hidden 1",
      nodes: [
        { value: "a_1", class: "fill-gray-300" },
        { value: "a_2", class: "fill-gray-300" },
        { value: "a_3", class: "fill-gray-300" },
      ],
    },
    {
      title: "Hidden 2",
      nodes: [
        { value: "a_1", class: "fill-gray-300" },
        { value: "a_2", class: "fill-gray-300" },
      ],
    },
  ];
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

  const x1 = new Value(5);
  x1._name = "Feature 1";
  const x2 = new Value(2);
  x2._name = "Feature 2";
  const w1 = new Value(1);
  w1._name = "Weight 1";
  const w2 = new Value(1);
  w2._name = "Weight 2";
  const w3 = new Value(1);
  w3._name = "Weight 3";
  const w4 = new Value(1);
  w4._name = "Weight 4";
  const w5 = new Value(1);
  w5._name = "Weight 5";
  const w6 = new Value(1);
  w6._name = "Weight 6";

  const s1 = w1.mul(x1);
  const s2 = w2.mul(x2);
  const s3 = w3.mul(x1);
  const s4 = w4.mul(x2);
  const a1 = s1.add(s2);
  a1._name = "a_1";
  const a2 = s3.add(s4);
  a2._name = "a_2";
  const s5 = a1.mul(w5);
  const s6 = a2.mul(w6);
  const o = s5.add(s6);
  o._name = "Output";
  o.backward();
</script>

<svelte:head>
  <title>Weight Initialization - World4AI</title>
  <meta
    name="description"
    content="Proper weight initialization techniques, like Xavier/Glorot or Kaiming/He initialization, can decrease the chances of exploding or vanishing gradients. Deep learning frameworks like PyTorch or Keras provide initializatin techniques out of the box."
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
  <NeuralNetwork {layers} padding={{ left: 0, right: 10 }} />
  <p>
    We will make the simplifying assumption, that there are no activation
    functions in the neural network and that we want to minimize the value of
    the output neuron and not some loss function. These assumptions do not have
    any effect on the results, but simplify notation and the depiction of the
    computational graph.
  </p>
  <p>
    If we have the same weight <Latex>w</Latex> for all nodes and layers, then in
    the very first forward pass all the neurons from the same layer will produce
    the same value.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
    a_1 &= x_1 * w_1 + x_2 * w_2 \\
    a_2 &= x_1 * w_3 + x_2 * w_4 \\
    a_1 &= a_2 
  \end{aligned}
    `}</Latex
  >
  <p>
    When we apply backpropagation we will quickly notice that the gradients with
    respect to the weights of the same feature are identical in each node.
  </p>
  <Latex
    >{String.raw`
    \dfrac{\partial o}{\partial a_1}
    \dfrac{\partial a_1}{\partial w_1}
    =
    \dfrac{\partial o}{\partial a_2}
    \dfrac{\partial a_2}{\partial w_3}
    `}</Latex
  >
  <p>
    The same starting values and the same gradients can only mean that all nodes
    in a layer will always have the same value. This is no different than having
    a neural network with a single neuron per layer. The network will never be
    able to solve complex problems. And if you initialize all your weights with
    zero, the network will always have dead neurons, always staying at the 0
    value.
  </p>
  <Alert type="danger">
    Never initialize your weights uniformly. Break the symmetry!
  </Alert>
  <p>
    Now let's use the same neural network and actually work though a dummy
    example. We assume feature values of 5 and 2 respectively and initialize all
    weights to 1.
  </p>
  <BackpropGraph graph={o} maxWidth={900} width={1180} height={920} />
  <p>
    Essentially you can observe two paths (left path and right path) in the
    computational graph above, representing the two neurons. But the paths are
    identical in their values and in their gradients. Even though there are 6
    weights in the neural network, half of them are basically clones.
  </p>
  <p>
    In order to break the symmetry researchers used to apply either a normal
    distribution (e.g. <Latex>\mu = 0</Latex> and <Latex>\sigma = 0.1</Latex>)
    or a uniform distribution (e.g in the range <Latex
      >{String.raw`-0.5 \text{ to } 0.5`}</Latex
    >) to initialize weights. This might seem reasonable, but Glorot and Bengio<InternalLink
      id={1}
      type="reference"
    /> showed that it is much more preferable to initialize weights based on the
    number of neurons that are used as input into the layer and the number of neurons
    that are inside a layer. This initializiation technique makes sure, that during
    the forward pass the variance of neurons stays similar from layer to layer and
    during the backward pass the gradients keep a constant variance from layer to
    layer. That condition reduces the likelihood of vanishing or exploding gradients.
    The authors proposed to initialize weights either using a uniform distribution
    <Latex>{String.raw`\mathcal{U}(-a, a)`}</Latex> where <Latex
      >{String.raw`a = \sqrt{\dfrac{6}{fan_{in} + fan_{out}}}`}</Latex
    > or the normal distribution <Latex
      >{String.raw`\mathcal{N}(0, \sigma^2)`}</Latex
    >, where
    <Latex>{String.raw`\sigma = \sqrt{\dfrac{2}{fan_{in} + fan_{out}}}`}</Latex
    >.
  </p>
  <p>
    The words <Latex>{String.raw`fan_{in}`}</Latex> and <Latex
      >{String.raw`fan_{out}`}</Latex
    > stand for the number of neurons that go into the layer as input and the number
    of neurons that are in the layer respectively. In the below example in the first
    hidden layer <Latex>{String.raw`fan_{in}`}</Latex> would be 2 and <Latex
      >{String.raw`fan_{out}`}</Latex
    > would be 3 respectively. In the second hidden layer the numbers would be exactly
    the other way around.
  </p>
  <NeuralNetwork
    layers={layers2}
    padding={{ left: 0, right: 20 }}
    height={100}
  />
  <p>
    While the Xavier/Glorot initialization was studied in conjunction with the
    sigmoind and the tanh activation function, the Kaiming/He initialization was
    designed to work with the ReLU activation<InternalLink
      id="2"
      type="reference"
    />. This is the standard initialization mode used in PyTorch.
  </p>
  <Alert type="info">
    For the most part you will not spend a lot of time dealing with weight
    initializations. Libraries like PyTorch and Keras have good common sense
    initialization values and allow you to switch between the initialization
    modes relatively easy. You do not nead to memorize those formulas. If you
    implement backpropagation on your own don't forget to at least break the
    symmetry.
  </Alert>
  <p>
    Implementing weight initialization in PyTorch is a piece of cake. PyTorch
    provides in <code>nn.init</code> different functions that can be used to
    initialize a tensor. You should have a look at the official
    <a
      href="https://pytorch.org/docs/stable/nn.init.html"
      target="_blank"
      rel="noreferrer"
    >
      PyTorch documentation</a
    > if you would like to explore more initialization schemes. Below for example
    we use the Kaiming uniform initialization on an empty tensor. Notice that the
    initialization is done inplace.
  </p>
  <PythonCode
    code={`W = torch.empty(5, 5)
# initializations are defined in nn.init
nn.init.kaiming_uniform_(W)`}
  />
  <PythonCode
    code={`tensor([[-0.9263, -0.0416,  0.0063, -0.8040,  0.8433],
        [ 0.3724, -0.9250, -0.2109, -0.1961,  0.3596],
        [ 0.6127,  0.2282,  0.1292,  0.8036,  0.8993],
        [-0.3890,  0.8515,  0.2224,  0.6172,  0.0440],
        [ 1.0282, -0.7566, -0.0305, -0.4382, -0.0368]])
`}
    isOutput={true}
  />
  <p>
    When we use the <code>nn.Linear</code> module, PyTorch automatically initializes
    weights and biases using the Kaiming He uniform initialization scheme. The sigmoid
    model from the last section was suffering from vanishing gradients, but we might
    remedy the problem, by changing the weight initialization. The Kaiming|He initialization
    was developed for the ReLU activation function, while the Glorot|Xavier initialization
    should be used with sigmoid activation functions. We once again create the same
    model that uses sigmoid activation functions. Only this time we loop over weights
    and biases and use the Xavier uniform initialization for weights and we set all
    biases to 0 at initialization.
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
        
        self.reset_parameters()
        
    def reset_parameters(self):
        with torch.inference_mode():
            for param in self.parameters():
                if param.ndim > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.zero_()
            
    def forward(self, features):
        return self.layers(features)`}
  />
  <PythonCode
    code={`model = SigmoidModel().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=ALPHA)`}
  />
  <p>This time around the model performs much better.</p>
  <PythonCode
    code={`history = train(train_dataloader, val_dataloader, model, criterion, optimizer)`}
  />
  <PythonCode
    code={`Epoch: 1/30 | Train Loss: 2.308537656906164 | Val Loss: 2.308519915898641
Epoch: 10/30 | Train Loss: 0.20434634447115443 | Val Loss: 0.2530187802116076
Epoch: 20/30 | Train Loss: 0.07002673296947375 | Val Loss: 0.15957480862239998
Epoch: 30/30 | Train Loss: 0.04259131510365781 | Val Loss: 0.1637446080291023
`}
    isOutput={true}
  />
  <PythonCode code={`plot_history(history, "xavier_metrics")`} />
  <img
    src={xavierMetrics}
    alt="Performance metrics with Xavier initialization"
  />
  <p>
    You should still use the ReLU activation function for deep neural networks,
    but be aware that weight initialization might have a significant impact on
    the performance of your model.
  </p>
</Container>
<Footer {references} />
