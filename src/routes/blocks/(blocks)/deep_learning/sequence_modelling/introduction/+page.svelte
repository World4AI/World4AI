<script>
  import { tweened } from "svelte/motion";
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import StepButton from "$lib/button/StepButton.svelte";

  let shuffled = false;
  let sentence = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
  ];

  let order = [0, 1, 2, 3, 4, 5, 6, 7, 8];

  let tweenedOrder = tweened(order, {
    duration: 500,
  });

  function changeOrder() {
    let neworder;
    if (!shuffled) {
      neworder = [1, 2, 0, 8, 7, 6, 5, 4, 3];
    } else {
      neworder = order;
    }
    tweenedOrder.set(neworder);
    shuffled = !shuffled;
  }
</script>

<svelte:head>
  <title>Sequence Modelling - World4AI</title>
  <meta
    name="description"
    content="Sequence models are designed to work with sequential data. When we talk about sequential data, we mean data where the order matters. This includes sentences , time series data like stock prices, speach, video and so on."
  />
</svelte:head>

<h1>Sequence Modelling</h1>
<div class="separator" />
<Container>
  <p>
    This chapter is dedicated to sequence modelling, a series of techniques that
    are very well suited to deal with <Highlight>sequential data</Highlight>.
  </p>
  <p>
    Most data that humans are dealing with and use for learning in their day to
    day life is sequential. The texts we are reading, the language we are
    hearing and the visual input we are processing are all sequential. A lot of
    structured data, like stock prices and weather data, also tends also to be
    sequential.
  </p>
  <Alert type="info">
    Sequential data is any type of data that needs to be organized in an ordered
    fashion (order matters) and there is most likely a correlation with previous
    data points.
  </Alert>
  <p>
    The feature of sequential data that separates it from other types of data is
    the importance of its order. When we were dealing with images of cats and
    dogs, our algorithm did not depend on the images to be sorted in any
    particular way. Sequential data on the other hand needs to be processed in a
    strictly sequential way.
  </p>
  <p>
    Look at the sentence below. You have probably seen this sentence before and
    it makes sense to you. If you interract with the example the sequence will
    shuffle.
  </p>
  <ButtonContainer>
    <StepButton on:click={changeOrder} />
  </ButtonContainer>
  <SvgContainer maxWidth={"100px"}>
    <svg viewBox="0 0 100 360">
      {#each $tweenedOrder as orderNumber, orderIdx}
        <g transform="translate(0 {5 + orderNumber * 40})">
          <rect
            class="fill-blue-100 stroke-black"
            x={1}
            y="0"
            width="90"
            height="30"
          />
          <text x={46} y={15}>{sentence[orderIdx]}</text>
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Does this new sequence still make sense to you? Probably not. Why should a
    neural network be able to work with a randomly shuffled sequence then?
  </p>
  <p>
    The feed forward neural networks that we have worked with so far do not take
    the sequence of the data into account. This chapter will therefore introduce
    a new type of a neural network that is very well suited for sequential data:
    a <Highlight>recurrent neural network</Highlight>.
  </p>
  <div class="separator" />
</Container>

<style>
  text {
    text-anchor: middle;
    dominant-baseline: middle;
    font-size: 20px;
  }
</style>
