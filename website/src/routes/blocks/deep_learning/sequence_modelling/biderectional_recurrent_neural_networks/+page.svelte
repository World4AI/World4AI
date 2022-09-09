<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Bidirectional Recurrent Neural Network</title>
  <meta
    name="description"
    content="Unlike a plain vanilla recurrent neural network, a biderectional rnn traverses the sequence in two directions. From front to back and from back to front. The output concatenates the two sets of hidden units. This architecture is especially well suited for translation task, as it becomes easier to determine context."
  />
</svelte:head>

<h1>Biderectional Recurrent Neural Networks</h1>
<div class="separator" />

<Container>
  <p>A recurrent neural network processes one part of the sequence at a time. When we are dealing with a sentence, the neural network starts with the very first letter and moves forward through the sentence. A biderectional recurrent neural network traverses the sequence from two directions. As usual from the start to finish and in the reverse direction, from finish to start. The output of the network, <Latex>y_t</Latex>, simply concatenates the two vectors that come from different directions.</p>
  <SvgContainer maxWidth={"800px"}>
    <svg viewBox="0 0 500 160">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="X {idx+1}" fontSize={12} x=50 y=150 width=25 height=15 color="var(--main-color-4)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color="var(--main-color-4)" />

          <!-- move in the forward direction -->
          <Arrow strokeWidth=1 data={[{x: 70, y:65}, {x:150, y: 65}] } />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-2)" />

          <!-- move in the reverse direction -->
          <Arrow strokeWidth=1 data={[{x: 150, y:85}, {x:70, y: 85}] } />
          <Block text="H_{4 - idx}" fontSize={12} x=95 y=85 width=25 height=15 color="var(--main-color-1)" />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>A biderectional recurrent neural network is especially well suited for translation tasks. Look at the two sentences below.</p>
  <p class="sentence">The bank opens ...</p>
  <p class="sentence">The bank of the river ...</p>
  <p>While the sentences start out with the same two words, the meaning can only be understood by reading through the whole sentence.</p>
  <p>But a biderectional RNN is not suited for every task. If you intend to predict future points of a time series data and the sequence contains that information somewhere in the sequence ahead, you will introduce data leakage.</p>
  <div class="separator" />
</Container>

<style>
  .sentence {
    background-color: var(--main-color-4);
    padding: 10px 0;
    text-align: center;
  }
</style>

