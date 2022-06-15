<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Graph from "./_graph/Graph.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Footer from "$lib/Footer.svelte";
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
    Instead, most practicioners would rather focus on the architecture of their
    neural network. Luckily when you use standard deep learning packackes like
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
  </p>
  <p>
    We are going to work through a simple example that has nothing to do with
    neural networks. This is done, because even a small neural network can
    produce a huge graph that would not fit on a website. We only expect you to
    gain some intuition regarding autodiff and the knowledge is transferable.
  </p>

  <p>Let us assume we are faced with the following function.</p>
  <Latex>{String.raw`f(x_1, x_2) = \ln(x_1) * \sin(x2) + e^{x_2}`}</Latex>
  <p>Our goal is to find the two partial derivatives:</p>
  <Latex>{String.raw`\dfrac{\partial f(x_1, x_2)}{\partial x_1}`}</Latex>,
  <Latex>{String.raw`\dfrac{\partial f(x_1, x_2)}{\partial x_2}`}</Latex>

  <p>
    While this is relatively trivial knowing the basic rules of calculus, we are
    going to utilize the graph representation of the equation, which looks as
    follows.
  </p>
  <Graph />
  <p>
    The yellow colored circles are inputs to the equation, the blue squares are
    the functions that process some inputs and the red sqaures are the outputs
    of some function. The lines represent what values are used as inputs into
    the function and what outputs are produced. We denote each of the outputs as <Latex
      >v</Latex
    > (for value).
  </p>
  <p>
    There are two modes of automatic differentiation: <Highlight
      >forward mode</Highlight
    > and <Highlight>reverse mode</Highlight>.
  </p>
  <p>
    Forward mode autodiff is inneficient for neural networks, because the
    complexity increases with the number of inputs. As there are thousands or
    even millions of weights and only one output (the loss function) the
    algorithm would require thousand or million passes.
  </p>
  <p>
    Backpropagation is a specific type of reverse mode autodifferentiation.
    Reverse mode autodiff can generally work with functions with several
    outputs, but in machine learning we usually deal with a single output, the
    loss. Reverse mode autodiff requires as many backward passes as there are
    outputs, but due to a single output we require just a single backward pass
    to calculate the gradients.
  </p>
  <p>
    We start the process by using concrete numbers for <Latex
      >{String.raw`x_1`}</Latex
    >
    <Latex>{String.raw`x_2`}</Latex> and save the intermediate values <Latex
      >{String.raw`v`}</Latex
    >, as those will be needed in the backward pass.
  </p>
  <p>
    Afterwards we calculate partial derivatieves from the end of the graph and
    apply the chain rule.
  </p>
  <p>
    The first equation calculates the output of the overall equation <Latex
      >{String.raw`v_5 = v_4 + v_3 `}</Latex
    >. We calculate the partial derivative of <Latex>{String.raw`v_5`}</Latex> with
    respect to the inputs and get 1 in both cases:
    <Latex>{String.raw`\dfrac{\partial v_5}{\partial v_4} = 1 `}</Latex>,
    <Latex>{String.raw`\dfrac{\partial v_5}{\partial v_3} = 1 `}</Latex>.
  </p>
  <p>
    If we follow the left path further up we calculate partial derivatives for <Latex
      >{String.raw`v_4 = v_1 * v_2`}</Latex
    >
    and get
    <Latex>{String.raw`\dfrac{\partial v_4}{\partial v_1} = v_2 `}</Latex>,
    <Latex>{String.raw`\dfrac{\partial v_5}{\partial v_2} = v_1 `}</Latex>.
  </p>
  <p>
    Now we face three equations at the top of the graph <Latex
      >{String.raw`v_1 = \ln(x_1)`}</Latex
    >, <Latex>{String.raw`v_2 = \sin(x_2)`}</Latex> and <Latex
      >{String.raw`v_3 = e^{x_2}`}</Latex
    >. Once again we only need to calculate the partial derivatives for the
    atomic function and we get
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
    You can notice a couple of things at this point. First, when the graph
    branches downwards, we add the partial derivatives due to the multivariable
    chain rule. Secod, we do not recalculate <Latex>v</Latex>, because those
    have already been calculated in the forward pass.
  </p>
  <p>
    While the example that we used in this chapter is extremely trivial, the
    same approach can be used for highly complex neural networks, as long as we
    can calculate partial derivatives for those atomic functions.
  </p>
  <p>If there is one takeaway from this chapter, it is the following.</p>
  <p class="warning">
    Backpropagation can be efficiently implemented using automatic
    differentiation. Do not reimplement backpropagation from scratch, but use
    already existing autodiff packages. You will save a lot of time and
    headaches if you rely on mature implementations like <a
      target="_blank"
      href="https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"
      >autograd</a
    > from PyTorch.
  </p>

  <Footer {references} />
</Container>
