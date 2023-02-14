<script>
  import Container from "$lib/Container.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  let references = [
    {
      author: "Sergey Ioffe, Christian Szegedy",
      title:
        "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
      journal: "",
      year: "2005",
      pages: "",
      volume: "",
      issue: "",
    },
  ];

  let layers = [
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-white" },
        { value: "x_2", class: "fill-white" },
        { value: "x_3", class: "fill-white" },
      ],
    },
    {
      title: "Hidden Layer 1",
      nodes: [
        { value: "a_1", class: "fill-w4ai-yellow" },
        { value: "a_2", class: "fill-w4ai-yellow" },
        { value: "a_3", class: "fill-w4ai-yellow" },
      ],
    },
    {
      title: "Hidden Layer 2",
      nodes: [
        { value: "a_1", class: "fill-w4ai-yellow" },
        { value: "a_2", class: "fill-w4ai-yellow" },
        { value: "a_3", class: "fill-w4ai-yellow" },
      ],
    },
    {
      title: "Out",
      nodes: [
        { value: "a_1", class: "fill-white" },
        { value: "a_2", class: "fill-white" },
        { value: "a_3", class: "fill-white" },
      ],
    },
  ];

  let batch = [
    {
      title: "Net Input",
      nodes: [{ value: "\\mathbf{z}", class: "fill-w4ai-yellow" }],
    },
    {
      title: "Activations",
      nodes: [{ value: "\\mathbf{a}", class: "fill-w4ai-yellow" }],
    },
    {
      title: "Batch Norm",
      nodes: [{ value: "\\mathbf{\\bar{a}}", class: "fill-w4ai-yellow" }],
    },
  ];
  let batch2 = [
    {
      title: "Net Input",
      nodes: [{ value: "\\mathbf{z}", class: "fill-w4ai-yellow" }],
    },
    {
      title: "Batch Norm",
      nodes: [{ value: "\\mathbf{\\bar{z}}", class: "fill-w4ai-yellow" }],
    },
    {
      title: "Activations",
      nodes: [{ value: "\\mathbf{a}", class: "fill-w4ai-yellow" }],
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Batch Normalization</title>
  <meta
    name="description"
    content="Similar to feature scaling, batch normalization normalizes hidden layers, thereby speeding up the training process, reducing overfitting and decreasin the chances of vanishing gradients."
  />
</svelte:head>

<h1>Batch Normalization</h1>
<div class="separator" />

<Container>
  <p>
    In a previous chapter we have discussed feature scaling. Feature scaling
    only applies to the input layer, so should we try to scale the intermediary
    features that are produced by hidden units? Would that be in any form
    benefitiary for trainig?
  </p>
  <NeuralNetwork height={100} width={250} maxWidth={"500px"} {layers} />
  <p>
    Sergey Ioffe and Christian Szegedy answered the question with a definitive
    yes<InternalLink type="reference" id="1" />. When we add so called <Highlight
      >batch normalizaton</Highlight
    >
    to hidden features, we can speed up the training process significantly, while
    gaining other additional advantages.
  </p>
  <p>
    Consider a particular layer <Latex>l</Latex>, to which output we would like
    to apply batch normalization. Using a batch of data we calculate the mean <Latex
      >{String.raw`\mu_j`}</Latex
    > and the variance
    <Latex>{String.raw`\sigma_j^2`}</Latex> for each hidden unit <Latex>j</Latex
    > in the layer.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
  \begin{aligned}
    \mu_j &= \dfrac{1}{n}\sum_{i=1}^n a_j^{(i)} \\
    \sigma_j^2 &= \dfrac{1}{n}\sum_{i=1}^n (a_j^{(i)} - \mu_j)
  \end{aligned}
    `}</Latex
    >
  </div>
  <p>
    Given those parameters we can normalize the hidden features, using the same
    procedure we used for feature scaling.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\hat{a}_j^{(i)} = \dfrac{a_j^{(i)} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}`}</Latex
    >
  </div>
  <p>
    The authors argued that this normalization procedure might theoretically be
    detremental to the performance, because it might reduce the expressiveness
    of the neural network. To combat that they introduced an additional step
    that allowed the neural network to reverse the standardization.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\bar{a}_j^{(i)} = \gamma_j \hat{a}_j^{(i)} + \beta_j`}</Latex
    >
  </div>
  <p>
    The feature specific parameters <Latex>\gamma</Latex> and <Latex
      >\beta</Latex
    > are learned by the neural network. If the network decides to set <Latex
      >\gamma_j</Latex
    > to
    <Latex>\sigma_j</Latex> and <Latex>\beta_j</Latex> to <Latex>\mu_j</Latex> that
    essentially neutralizes the normalization. If normalization indeed worsens the
    performance, the neural network has the option to reverse the normalization step.
  </p>
  <p>
    Our formulations above indicated that batch normalization is applied to
    activations. This procedure is similar to input feature scaling, because you
    normalize the data that is processed in the next layer.
  </p>
  <NeuralNetwork
    height={50}
    width={250}
    maxWidth="600px"
    layers={batch}
    padding={{ left: 0, right: 30 }}
  />
  <p>
    In practice though batch norm is often applied to the net inputs and the
    result is forwarded to the activation function.
  </p>
  <NeuralNetwork
    height={50}
    width={250}
    maxWidth={"600px"}
    layers={batch2}
    padding={{ left: 0, right: 30 }}
  />
  <p>
    There is no real consensus about how you should apply batch normalization,
    but this decision in all likelihood should not make or break your project.
  </p>
  <p>
    There is an additional caveat. In practice we often remove the bias term <Latex
      >b</Latex
    > from the calculation of the net input <Latex
      >{String.raw`z = \mathbf{xw^T} + b`}</Latex
    >. This is done due to the assumption, that <Latex>\beta</Latex> essentially
    does the same operation. Both are used to shift the calculation by a constant
    and there is hardly a reason to do that calculation twice.
  </p>
  <p>
    The authors observed several adantages that batch normalization provides.
    For once batch norm makes the model less sensitive to the choice of the
    learning rate, which allows us to increase the learning rate and thereby
    increase the speed of convergence. Second, the model is more forgiving when
    choosing bad initial weights. Third, batch normalization seems to help with
    the vanishing gradients problem. Overall the authors observed a significant
    increase in training speed, thus requiring less epochs to reach the desired
    performance. Finally batch norm seems to act as a regularizer. When we train
    the neural network we calculate the mean <Latex>\mu_j</Latex> and the standard
    deviation <Latex>\sigma_j</Latex> one batch at a time. This calculation is noisy
    and the neural network has to learn to tune out that noise in order to achieve
    a reasonable performance.
  </p>
  <p>
    During inference the procedure of calculating per batch statistics would
    cause problems, because different inference runs would generate different
    means and standard deviations and therefore generate different outputs. We
    want the neural network to be deterministic during inference. The same
    inputs should always lead to the same outputs. For that reason during
    training the batch norm layer calculates a moving average of <Latex
      >\mu</Latex
    > and <Latex>\sigma</Latex> that can be used at inference time.
  </p>
  <p>
    Let also mention that no one really seems to know why batch norm works.
    Different hypotheses have been formulated over the years, but there seems to
    be no clear consensus on the matter. All you have to know is that batch
    normalization works well and is almost a requirement when training modern
    deep neural networks. This technique will become one of your main tools when
    designing modern neural network architectures.
  </p>
  <p>
    PyToch has an explicit <code>BatchNorm1d</code> module that can be applied
    to a flattened tensor, like the flattened MNIST image. The 2d version will
    become important when we start dealing with 2d images. Below we create a
    small module that combines a linear mapping, batch normalization and a
    non-linear activation. Notice that we we provide the linear module with the
    argument <code>bias=False</code> in order to deactivate the bias calculation.
  </p>
  <PythonCode
    code={`HIDDEN_FEATURES = 70
class BatchModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES, bias=False),
            nn.BatchNorm1d(HIDDEN),
            nn.ReLU()
        )
    
    def forward(self, features):
        return self.layers(features)`}
  />
  <p>We can reuse the above defined module several times.</p>
  <PythonCode
    code={`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(NUM_FEATURES, HIDDEN_FEATURES),
                BatchModule(),
                BatchModule(),
                BatchModule(),
                nn.Linear(HIDDEN, NUM_LABELS),
            )
    
    def forward(self, features):
        return self.layers(features)`}
  />
  <p>
    As the batch normalization layer behaves differently during training and
    evalutation, don't forget to switch between <code>model.train()</code>
    and <code>model.eval()</code>.
  </p>
</Container>
<Footer {references} />
