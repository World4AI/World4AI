<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
</script>

<svelte:head>
  <title>Logistic Regression - World4AI</title>
  <meta
    name="description"
    content="Logistic regression builds upon simple linear regression and can be either used separately for classification tasks or used as a unit in a neural network architecture."
  />
</svelte:head>

<h1>Logistic Regression</h1>
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
        <path
          stroke-width="2px"
          d="m220 80c10.001 0 20.001 0 25-9.9997s4.9995-29.999 10-40 15-10 24.999-10"
        />
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
    Linear regression can be regarded as the basis for <Highlight
      >logistic regression</Highlight
    >. Unlike the name suggests though, logistic regression is commonly used for
    classification and not regression tasks.
  </p>
  <Alert type="info">
    Logistic regression is a neuron that combines linear regression and a
    non-linear activation function.
  </Alert>
  <p>
    In this section we will discover that logistic regression can also be
    regarded as a non-linear neuron. The study of logistic regression should
    therefore provide us with the building blocks that will eventually allow us
    to construct fully functional neural networks.
  </p>
  <div class="separator" />
</Container>
