<script>
  import Container from "$lib/Container.svelte";
  import Network from "./_intro/Network.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Neural Network</title>
  <meta
    name="description"
    content="The training process of a neural network consists of a forward and a backward pass."
  />
</svelte:head>

<h1>Neural Network</h1>
<div class="separator" />
<Container>
  <p>
    Neural networks allow us to make all the marvelous calculations that lead to
    high performance in image classification, text generation, computer games
    and much more.
  </p>
  <p>
    The training process of neural networks is always done using the same
    procedure consisting of two major steps. In the <Highlight
      >forward pass</Highlight
    > a mini batch is used to calculate the loss of the neural network. In the <Highlight
      >backward pass</Highlight
    > the loss is distributed among the neurons and the proportional loss is used
    to adjust individual weights. The passes run sequentially, until some condition
    is met for the training to terminate.
  </p>
  <SvgContainer maxWidth="700px">
    <Network />
  </SvgContainer>
  <p>
    No matter how complicated the neural network looks like on the surface, the
    forward and the backward pass are always present in modern day deep
    learning.
  </p>
  <div class="separator" />
</Container>
