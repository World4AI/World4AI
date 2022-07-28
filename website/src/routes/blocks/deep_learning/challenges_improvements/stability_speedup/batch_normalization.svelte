<script>
  import Container from "$lib/Container.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

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
        { value: "x_1", fill: "none" },
        { value: "x_2", fill: "none" },
        { value: "x_3", fill: "none" },
      ],
    },
    {
      title: "Hidden Layer 1",
      nodes: [
        { value: "a_1", fill: "var(--main-color-3)" },
        { value: "a_2", fill: "var(--main-color-3)" },
        { value: "a_3", fill: "var(--main-color-3)" },
      ],
    },
    {
      title: "Hidden Layer 2",
      nodes: [
        { value: "a_1", fill: "var(--main-color-3)" },
        { value: "a_2", fill: "var(--main-color-3)" },
        { value: "a_3", fill: "var(--main-color-3)" },
      ],
    },
    {
      title: "Out",
      nodes: [
        { value: "a_1", fill: "none" },
        { value: "a_2", fill: "none" },
        { value: "a_3", fill: "none" },
      ],
    },
  ];

  let batch = [
    {
      title: "",
      nodes: [{ value: "\\mathbf{z}", fill: "var(--main-color-3)" }],
    },
    {
      title: "Activations",
      nodes: [{ value: "\\mathbf{a}", fill: "var(--main-color-3)" }],
    },
    {
      title: "Batch Norm",
      nodes: [{ value: "\\mathbf{\\bar{a}}", fill: "var(--main-color-3)" }],
    },
    {
      title: "",
      nodes: [{ value: "\\mathbf{z}", fill: "var(--main-color-3)" }],
    },
  ];
  let batch2 = [
    {
      title: "",
      nodes: [{ value: "\\mathbf{z}", fill: "var(--main-color-3)" }],
    },
    {
      title: "Batch Norm",
      nodes: [{ value: "\\mathbf{\\bar{z}}", fill: "var(--main-color-3)" }],
    },
    {
      title: "Activations",
      nodes: [{ value: "\\mathbf{a}", fill: "var(--main-color-3)" }],
    },
    {
      title: "",
      nodes: [{ value: "\\mathbf{z}", fill: "var(--main-color-3)" }],
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
    In a previous section we have discussed the need to scale the input features
    in order to speed up training of a neural network. And while standardizing
    or normalizing the input features can speed up the training process
    significantly, it makes sense to ask ourselves the following question.
    Should we try to scale the intermediary features that come out of hidden
    units? Would that be in any form benefitary for trainig?
  </p>
  <NeuralNetwork height={100} width={250} maxWidth={"700px"} {layers} />
  <p>
    Sergey Ioffe and Christian Szegedy answered the question with a definitive
    yes<InternalLink type="reference" id="1" />. When we add so called <Highlight
      >batch normalizaton</Highlight
    >
    to hidden features, we can speed up the training process significantly, while
    gaining other additional advantages.
  </p>
  <p>
    Consider a particular layer <Latex>l</Latex> to which output we would like to
    apply batch normalization. With each new batch for each hidden feature <Latex
      >j</Latex
    > we calculate the mean
    <Latex>{String.raw`\mu_j`}</Latex> and the variance
    <Latex>{String.raw`\sigma_j^2`}</Latex>.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
    \mu_j &= \dfrac{1}{n}\sum_{i=1}^n a_j^{(i)} \\
    \sigma_j^2 &= \dfrac{1}{n}\sum_{i=1}^n (a_j^{(i)} - \mu_j)
  \end{aligned}
    `}</Latex
  >
  <p>Given those parameters we normalize the hidden features.</p>
  <Latex
    >{String.raw`\hat{a}_j^{(i)} = \dfrac{a_j^{(i)} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}`}</Latex
  >
  <p>
    The authors argued that this normalization procedure might theoretically be
    detremental to the performance, because it might change what the layer is
    able represent. To combat that they introduced an additional step that
    allowed the neural network to reverse the standardization.
  </p>
  <Latex
    >{String.raw`\bar{a}_j^{(i)} = \gamma_j \hat{a}_j^{(i)} + \beta_j`}</Latex
  >
  <p>
    The feature specific parameters <Latex>\gamma</Latex> and <Latex
      >\beta</Latex
    > are learned by the neural network. If the network decides to set <Latex
      >\gamma_j</Latex
    > to
    <Latex>\sigma_j</Latex> and <Latex>\beta_j</Latex> to <Latex>\mu_j</Latex> that
    essentially neutralizes the normalization. If normalization indeed worsens the
    performance, the neural network has the option to reverse the normalization and
    thereby to produce the identity function.
  </p>
  <p>
    Our formulations above indicated that batch normalization is applied to
    activations. This procedure is similar to input feature scaling, because you
    normalize the data to be processed in the next layer.
  </p>
  <NeuralNetwork height={50} width={250} maxWidth={"700px"} layers={batch} />
  <p>
    In practice though batch norm is often applied to the net inputs and the
    result is forwarded to the activation function.
  </p>
  <NeuralNetwork height={50} width={250} maxWidth={"700px"} layers={batch2} />
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
    increase the speed of learning. Second, the model is more forgiving when
    choosing bad initial weights and seems to help with the vanishing gradients
    problem. Overall the authors observed a significant increase in training
    speed, requiring less epochs to arrive at the same performance. Finally
    batch norm seems to act as a regularizer. When we train the neural network
    we calculate the mean <Latex>\mu_j</Latex> and the standard deviation <Latex
      >\sigma_j</Latex
    > one batch at a time. This calculation is noisy and the neural network has to
    learn to tune out that noise in order to achieve a reasonable performance. During
    inference this procedure would cause problems, because different inference runs
    would create different batches and therefore generate different outputs. But
    we want the neural network to be deterministic during inference. The same inputs
    should always lead to the same outputs. For that reason during training the batch
    norm layer calculates a moving average of <Latex>\mu</Latex> and <Latex
      >\sigma</Latex
    > that can be used at inference time.
  </p>
  <p>
    Let us finish this chapter by mentioning that no one really seems to know
    why batch norm works. Different hypotheses have been formulated over the
    years, but there seems to be no clear consensus on the matter. All you have
    to know is that batch normalization works well and is almost a requirement
    when training modern deep neural networks. This technique will become one of
    your main tools when designing modern neural network architectures.
  </p>
</Container>
<Footer {references} />
