<script>
  import { tweened } from "svelte/motion";
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import StepButton from "$lib/button/StepButton.svelte";
  import Latex from "$lib/Latex.svelte";

  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

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
    Most data that humans are dealing with and use for learning in their day to
    day life is sequential. The texts we are reading, the language we are
    hearing and the visual input we are processing are all sequential. A lot of
    structured data, like stock prices and weather data, also tends to be
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
    Look at the sentence below for example. You have probably seen this sentence
    before and it should make sense to you. If you interract with the example
    the sequence will shuffle.
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
    In this chapter we are going to focus on <Highlight
      >sequence modelling</Highlight
    >, a series of techniques that are very well suited to deal with sequential
    data. Especially we will focus on so called <Highlight
      >autoregressive models</Highlight
    >.
  </p>
  <Alert type="info"
    >An autoregressive model uses past values of the sequence to predict the
    next value in the sequence.</Alert
  >
  <p>
    We could use an autoregressive model for example to predict the fifth word
    in a sentence, given the previous four words.
  </p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 275 150">
      {#each Array(5) as _, idx}
        <Block
          width={30}
          height={30}
          x={17 + 60 * idx}
          y={130}
          text="x_{idx + 1}"
          type={"latex"}
          fontSize={15}
          class={idx !== 4 ? "fill-sky-100" : "fill-red-400"}
        />
        {#if idx !== 4}
          <Arrow
            strokeWidth={1.5}
            dashed={true}
            strokeDashArray="4 4"
            moving={true}
            data={[
              { x: 17 + 60 * idx, y: 115 },
              { x: 17 + 60 * idx, y: 10 + 30 * idx },
              { x: 28 + 60 * 4 - idx * 8, y: 10 + 30 * idx },
              { x: 28 + 60 * 4 - idx * 8, y: 110 },
            ]}
          />
        {/if}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Mathematically we can express this idea as the probability of a value in a
    sequence, given the previous values: <Latex
      >{String.raw`P(x_t | x_{t-1}, x_{t-2}, \dots, x_1 )`}</Latex
    >.
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
