<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Linear Neuron</title>
  <meta
    name="description"
    content="Linear regression constitutes the simplest neuron imaginable: a neuron with a linear activation function."
  />
</svelte:head>

<Container>
  <h1>Linear Neuron</h1>
  <div class="separator" />
  <p>
    If we look at linear regression from the perspective of a neural network, we
    will observe that linear regression is just a neuron with a linear
    activation function.
  </p>

  <p>
    Let us remind ourselves, that a neuron is a computational unit. The output
    of the neuron is based on three distinct calculations: scaling of inputs <Latex
      >{String.raw`\mathbf{x}`}</Latex
    > with weights <Latex>{String.raw`\mathbf{w}`}</Latex>, summation of the
    scaled inputs (plus bias <Latex>b</Latex>) and the application of an
    activation function.
  </p>
  <p>
    The dot product

    <Latex>{String.raw`\mathbf{w}\cdot\mathbf{x} + b`}</Latex>
    essentially applies all three parts of the calculation in a single step.
  </p>
  <Latex
    >{String.raw`
\large
\begin{bmatrix}
w_1 \\
w_2 \\
w_3 \\
\vdots \\
w_n \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_n \\
\end{bmatrix}
+ b
=
\begin{bmatrix}
w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n + b
\end{bmatrix}
`}</Latex
  >
  <p>
    At this point you might interject, that we do not have an activation
    function, so let us introduce one that does not change the nature of linear
    regression. We are going to use the so called identity function where input
    equals output <Latex>I(x) = x</Latex>. When we apply the identity function
    as an activation we end up with a linear neuron, where

    <Latex>{String.raw`y = I(\mathbf{w}\cdot\mathbf{x} + b)`}</Latex>. This
    might seem like an unnecessary step, but by enforcing the usage of an
    identity function, we put ourselves into a position where we can start to
    understand different types of neurons. All we have to do is to replace the
    identity function <Latex>I(x)</Latex> by any other activation function <Latex
      >a(x)</Latex
    > to describe any other type of neuron:
    <Latex>{String.raw`y = a(\mathbf{w}\cdot\mathbf{x} + b)`}</Latex>.
  </p>
  <div class="separator" />
</Container>
