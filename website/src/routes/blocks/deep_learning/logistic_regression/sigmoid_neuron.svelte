<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Sigmoid Neuron</title>
  <meta
    name="description"
    content="Logistic regression constitutes the simplest non-linear neuron: a neuron with the sigmoid activation function."
  />
</svelte:head>

<h1>Sigmoid Neuron</h1>
<div class="separator" />
<Container>
  <p>
    Let us make the same exercise we did with linear regression. We can look at
    logistic regression from the perspective of a neural network. If we do, we
    will observe that logistic regression is a neuron with a sigmoid activation
    function.
  </p>

  <p>
    Once again let us remind ourselves, that a neuron is a computational unit
    that is based on three distinct steps. First: the inputs <Latex
      >{String.raw`\mathbf{x}`}</Latex
    > are scaled by weights <Latex>{String.raw`\mathbf{w}`}</Latex>. Second: the
    scaled inputs (plus bias <Latex>b</Latex>) are aggregated via a sum. Third:
    an activation function <Latex>a</Latex> is applied to the sum.
  </p>
  <p>
    In logistic regression all three steps can be described by
    <Latex>{String.raw`\sigma(\mathbf{w^T}\mathbf{x} + b)`}</Latex>, where <Latex
      >\sigma</Latex
    > is the sigmoid activation function. Written in a more familiar manner the output
    of the neuron amounts to: <Latex
      >{String.raw`\dfrac{1}{1+e^{-(w_1x_1 + \cdots + w_nx_n + b)}}`}</Latex
    >.
  </p>
  <p>
    This type of a neuron is extremely powerful. When we combine different
    sigmoid neurons, such that the output of a neuron is used as an input to the
    neurons in the next layer, we essentially create a neural network.
  </p>
  <div class="separator" />
</Container>
