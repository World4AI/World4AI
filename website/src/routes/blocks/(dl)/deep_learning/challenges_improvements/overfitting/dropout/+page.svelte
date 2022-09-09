<script>
  import Container from "$lib/Container.svelte";
  import Dropout from "../_dropout/Dropout.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";

  const references = [
    {
      author:
        "G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever and R. R. Salakhutdivno",
      title:
        "Improving neural networks by preventing co-adaptation of feature detectors",
      journal: "",
      year: "2012",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author:
        "Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov",
      title:
        "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
      journal: "Journal Of Machine Learning Research",
      year: "2014",
      pages: "1929-1958",
      volume: "15",
      issue: "1",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Dropout</title>
  <meta
    name="description"
    content="Dropout is a regularization technique that works by removing different activation nodes at each training step."
  />
</svelte:head>

<h1>Dropout</h1>
<div class="separator" />

<Container>
  <p>
    Dropout is a regularization technique was developed by Geoffrey Hinton and
    his colleagues at the university of Toronto <InternalLink
      id={1}
      type="reference"
    />
    <InternalLink id={2} type="reference" />. The idea seems so simple, that it
    is almost preposterous that it works at all.
  </p>
  <p>
    At trainig time at each training step with a probability of <Latex>p</Latex>
    a neuron is deactivated, its value is set to 0. You can start the interactive
    example and observe how it looks like with a <Latex>p</Latex> value of 0.2.
  </p>
  <Dropout />
  <p>
    When we use our model for inference, we do not remove any of the nodes. If
    we did that, we would get different results each time we run a model. By not
    deactivating any nodes we introduce a problem though. Because more neurons
    are active, each layer has to deal with an input that is on a different
    scale, than the one the neural network has seen during training. Different
    conditions during training and inference will prevent the neural network
    from generating good predictions. To prevent that from happening the active
    nodes are scaled by
    <Latex>{String.raw`\dfrac{1}{1-p}`}</Latex> during training. We skip that scaling
    during inference and thus create similar conditions for training and inference.
  </p>
  <p>
    Let us assume for example that the activation layer contains only 1's and
    <Latex>p</Latex> is 0.5.
  </p>
  <Latex
    >{String.raw`
    \begin{bmatrix}
    1 \\
    1 \\
    1 \\
    1 \\
    1 \\
    1 \\
    \end{bmatrix}
    `}</Latex
  >
  <p>
    The dropout layer will zero out roughly half of the activations and multiply
    the rest by <Latex>{String.raw`\dfrac{1}{1-0.5} = 2`}</Latex>.
  </p>
  <Latex
    >{String.raw`
    \begin{bmatrix}
    2 \\
    0 \\
    0 \\
    2 \\
    0 \\
    2 \\
    \end{bmatrix}
    `}</Latex
  >
  <p>
    But why is the dropout procedure helpful in avoiding overfitting? Each time
    we remove a set of activations from training, we essentially create a
    different model. This simplified model has to learn to deal with the task at
    hand without overrelying on any of the previous activations, because any of
    those might get deactivated at any time. The final model can be seen as an
    ensemble of an immensely huge collection of simplified models. Ensembles
    models tend to produce better results and reduce overfitting. You will
    notice that in practice dropout works extremely well.
  </p>
</Container>

<Footer {references} />
