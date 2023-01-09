<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Alert from "$lib/Alert.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>Linear Regression Introduction - World4AI</title>
  <meta
    name="description"
    content="Linear regression is the simplest neuron that we can come up with. The study of linear regression makes the further study of neural networks more convenient."
  />
</svelte:head>

<h1>Linear Regression</h1>
<div class="separator" />
<Container>
  <SvgContainer maxWidth={"500px"}>
    <svg version="1.1" viewBox="0 0 500 100" xmlns="http://www.w3.org/2000/svg">
      <g fill="none" stroke="#000">
        <rect
          x="210"
          y="10"
          width="80"
          height="80"
          stroke-linecap="square"
          stroke-linejoin="round"
          class="fill-w4ai-yellow"
        />
        <path stroke-width="2px" d="m220 80 60-60" />
        <g stroke-width="2px">
          <path d="m20 20h170" />
          <path d="m20 50h170" />
          <path d="m20 80h170" />
          <path d="m310 50h170" />
        </g>
      </g>
    </svg>
  </SvgContainer>
  <p>
    <Highlight>Linear regression</Highlight> is hardly used in practice for problems
    with huge datasets or problems that show complex relationships between the features
    and the targets. So what is the purpose of studying linear regression when neural
    networks produce much better results? Linear regression is a precursor to neural
    networks.
  </p>
  <Alert type="info"
    >Linear regression models can be interpreted as very simple neurons in a
    neural network.</Alert
  >
  <p>
    Linear regression models are essentially the simplest neurons we can think
    of, therefore it is convenient to study the basic concepts that are used in
    all neural networks by studying linear regression. When we finally arrive at
    more complex deep neural networks, we will be familiar with the basic
    building blocks of neurons.
  </p>
  <Alert type="warning">
    It is ok if at this point you do not yet understand the relationship between
    a neural network and the linear regression. Most likely you will need to go
    through the next three chapters to fully grasp the concept. Return here
    after you have finished the neural networks chapter to see if our definition
    of linear regression as a neuron makes sense to you.
  </Alert>
  <div class="separator" />
</Container>
