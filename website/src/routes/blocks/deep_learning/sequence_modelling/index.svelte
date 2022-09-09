<script>
  import { tweened } from 'svelte/motion';
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";

  let sentence = [
    "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
  ]
  
  let order = [0, 1, 2, 3, 4, 5, 6, 7, 8];

  let tweenedOrder = tweened(order, {
    duration: 500
  });

  function changeOrder() {
    //order.sort(() => Math.random() - 0.5);    
    let neworder = [1, 2, 0, 8, 7, 6, 5, 4, 3];
    tweenedOrder.set(neworder); 
  }

</script>

<svelte:head>
  <title>World4AI | Deep Learning | Sequence Modelling</title>
  <meta
    name="description"
    content="Sequential models are designed to work with sequential data, data where the order matters."
  />
</svelte:head>

<h1>Sequence Modelling</h1>
<div class="separator" />

<Container>
  <p>This chapter is dedicated to sequence modelling, a series of techniques that are very well suited to deal with <Highlight>sequential data</Highlight>.</p>
  <p>Most data that humans are dealing with and use for learning in their day to day life is sequential. The texts we are reading, the language we are hearing and the visual input we are processing are all sequential. A lot of structured data, like stock prices and weather data, also tends also to be sequential.</p>
  <p class="info">Sequential data is any type of data that needs to be organized in an ordered fashion (order matters) and there is most likely a correlation with previous data points.</p>
  <p>The feature of sequential data that separates it from other types of data is the importance of its order. When we were dealing with images of cats and dogs, our algorithm did not depend on the images to be sorted in any particular way. Sequential data on the other hand needs to be processed in a strictly sequential way.</p> 
  <p>Look for example at the below sentence. You have probably seen this sentence before and it makes sense to you. If you click on any of the words the sequence will reshuffle.</p>
  <svg viewBox="0 0 630 100">
    {#each $tweenedOrder as orderNumber, orderIdx}
      <g transform="translate({5 + orderNumber*70} 0)">
        <rect on:click={changeOrder} x={0} y=40 width=60 height=30 fill="var(--main-color-2)" stroke="black" />
        <text x={30} y={40 + 15}>{sentence[orderIdx]}</text>
      </g>
    {/each}
  </svg>
  <p>Does this sequence still make sense to you? Probably not. Why should a neural network work with a randomly shuffled sequence then?</p>
  <p>The feed forward neural networks that we have worked with so far do not take the sequence of the data into account. This chapter will therefore introduce a new type of a neural network that is very well suited for sequential data: a recurrent neural network.</p>
  <div class="separator"/>
</Container>

<style>
  rect {
    cursor: pointer;
  }

  text {
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
  }
</style>
