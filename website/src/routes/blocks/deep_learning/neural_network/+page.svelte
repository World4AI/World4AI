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
    We can train neural networks that can classify images, generate text or play
    computer games. No matter what task we are trying to accomplish, the
    training process of neural networks is always done using the same procedure
    consisting of two major steps: the <Highlight>forward pass</Highlight> and the
    <Highlight>backward pass</Highlight>. In the forward pass a mini batch is
    used to calculate the loss of the neural network. In the backward pass the
    loss is distributed among all neurons, from the last neurons to the first
    neurons, and the weights and biases are adjusted proportional to their
    contribution to the loss. The forward and the backward pass run
    back-to-back, until some condition is met and the training process is
    terminated.
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
