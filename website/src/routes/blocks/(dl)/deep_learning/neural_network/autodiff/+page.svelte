<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Graph from "../_graph/Graph.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Footer from "$lib/Footer.svelte";
  import Alert from "$lib/Alert.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  let references = [
    {
      author:
        "Atılım Güneş Baydin,  Barak A. Pearlmutter,  Alexey Andreyevich Radul,  Jeffrey Mark Siskind",
      title: "Automatic differentiation in machine learning: a survey",
      journal: "The Journal of Machine Learning Research",
      year: "2017",
      pages: "5595-5637",
      volume: "18",
      issue: "1",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Autodiff</title>
  <meta
    name="description"
    content="Automatic differentiation (autodiff) is a technique used by all modern deep learning libraries to calculate the gradients. Backpropagation is essentially a special case of a reverse mode autodiff with just one output."
  />
</svelte:head>

<h1>Automatic Differentiation</h1>
<div class="separator" />
<Container>
  <p>
    In practice it would be highly inconvenient and prone to errors to reinvent
    the wheel from scratch and to reimplement backpropagation again and again.
    Instead, most practicioners would rather focus on the problem they are
    trying to solve. Luckily when you use standard deep learning packackes like
    PyTorch or TensorFlow, backpropagation is already an integral part of those
    libraries. For the most part we can get away with not knowing how exactly
    backpropagation is implemented internally, but we will cover at least some
    basics, to understand some of the jargon. If you want to take a deep dive on
    your own, you can look at Baydin et. al.<InternalLink
      type="reference"
      id="1"
    />
  </p>
  <p>
    The above mentioned packages use a technique called <Highlight
      >automatic differentiation</Highlight
    > or <Highlight>autodiff</Highlight> to calculate the gradients of a particular
    function. Autodiff constructs a so called <Highlight
      >computational graph</Highlight
    > that tracks all the basic/atomic calculations of a much more complex function.
    The autodiff package has internal knowledge of how to calculate the derivatives
    of those simple functions like additions, multiplications, logarithms and exponentials.
    Those atomic functions are used to construct a more complex function that we
    are interested in, like the model of a neural network. The gradients of the complex
    function can then be calculated by combining the gradients of the atomic functions
    using basic rules of calculus, like the product rule, the sum rule and especially
    the chain rule.
  </p>
  <p>
    There are generally two modes of automatic differentiation: <Highlight
      >forward mode</Highlight
    > and <Highlight>reverse mode</Highlight>.
  </p>
  <p>
    Forward mode autodiff gets inneficient when we deal with neural networks,
    because the number of passes that is required to calculate the cradients
    corresponds to the number of inputs. There are thousands or even millions of
    weights and biases in a neural network, therefore the algorithm would
    require thousand or million passes through a highly complex function.
  </p>
  <p>
    Reverse mode autodiff requires as many backward passes through a function,
    as there are outputs produced by that function. Backpropagation is a
    specific type of reverse mode autodifferentiation, because in machine
    learning we usually deal with a single output, the loss. For that reason we
    require single backward pass to calculate the gradients.
  </p>
  <p>
    We are going to work through a simple example that has nothing to do with
    neural networks. This is done, because even a small neural network can
    produce a huge graph that would not fit on a website. We only expect you to
    gain some intuition regarding autodiff and a simple function should do the
    trick.
  </p>

  <p>Let us assume we face the following function.</p>
  <Latex>{String.raw`f(x_1, x_2) = \ln(x_1) * \sin(x2) + e^{x_2}`}</Latex>
  <p>Our goal is to find the two partial derivatives:</p>
  <Latex>{String.raw`\dfrac{\partial f(x_1, x_2)}{\partial x_1}`}</Latex>,
  <Latex>{String.raw`\dfrac{\partial f(x_1, x_2)}{\partial x_2}`}</Latex>

  <p>
    While this is relatively trivial knowing the basic rules of calculus, we are
    going to utilize the graph representation of the same equation.
  </p>
  <Graph />
  <p>
    The yellow colored circles are inputs to the equation, the blue squares are
    the operations that process the inputs and the red sqaures are the
    intermediary outputs. The lines show the flow of data from the yellow inputs
    at the top to the final output <Latex>v_5</Latex>. We denote each of the
    outputs as <Latex>v</Latex> (for value).
  </p>
  <p>
    In the backward pass we start with the output of the function: <Latex
      >{String.raw`v_5 = v_4 + v_3 `}</Latex
    >. We calculate the partial derivative of <Latex>{String.raw`v_5`}</Latex> with
    respect to the inputs and get 1 in both cases:
    <Latex>{String.raw`\dfrac{\partial v_5}{\partial v_4} = 1 `}</Latex>,
    <Latex>{String.raw`\dfrac{\partial v_5}{\partial v_3} = 1 `}</Latex>.
  </p>
  <p>
    If we follow the left path, we encounter <Latex
      >{String.raw`v_4 = v_1 * v_2`}</Latex
    >. The partial derivatives are easily obtained using the product rule:
    <Latex>{String.raw`\dfrac{\partial v_4}{\partial v_1} = v_2 `}</Latex>,
    <Latex>{String.raw`\dfrac{\partial v_5}{\partial v_2} = v_1 `}</Latex>.
  </p>
  <p>
    Now we face three equations at the top of the graph <Latex
      >{String.raw`v_1 = \ln(x_1)`}</Latex
    >, <Latex>{String.raw`v_2 = \sin(x_2)`}</Latex> and <Latex
      >{String.raw`v_3 = e^{x_2}`}</Latex
    >. Once again we only need to calculate the partial derivatives for the
    atomic functions and we get:
    <Latex
      >{String.raw`\dfrac{\partial v_1}{\partial x_1} = \dfrac{1}{x_1}`}</Latex
    >,
    <Latex>{String.raw`\dfrac{\partial v_2}{\partial x_2} = \cos(x_2)`}</Latex>,
    and
    <Latex>{String.raw`\dfrac{\partial v_3}{\partial x_2} = e^{x_2}`}</Latex>.
  </p>
  <p>
    The chain rule is applied at each step of the backward process and we end up
    with the following partial derivatives.
  </p>
  <Latex
    >{String.raw`\dfrac{\partial v_5}{\partial x_1} = 1 * v_2 * \dfrac{1}{x_1} \\ `}</Latex
  >
  <br />
  <Latex
    >{String.raw`\dfrac{\partial v_5}{\partial x_2} = 1 * v_1 * cos(x_2) + 1 * e^{x_2}`}</Latex
  >
  <p>
    While the example that we used in this chapter is extremely trivial, the
    same approach can be used for highly complex neural networks, as long as we
    can calculate partial derivatives for those atomic functions.
  </p>

  <Alert type="warning">
    Backpropagation can be efficiently implemented using automatic
    differentiation. Do not reimplement backpropagation from scratch, but use
    already existing autodiff packages. You will save a lot of time and
    headaches if you rely on mature implementations like <a
      target="_blank"
      href="https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"
      >autograd</a
    > from PyTorch.
  </Alert>

  <Footer {references} />
</Container>
