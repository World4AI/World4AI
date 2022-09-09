<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Plus from "$lib/diagram/Plus.svelte";
  import Circle from "$lib/diagram/Circle.svelte";
  import Border from "$lib/diagram/Border.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Types of Recurrent Neural Networks</title>
  <meta
    name="description"
    content="There are different types of recurrent neural network, like seq-to-seq, seq-to-vec, vec-to-seq and encoder-decoder."
  />
</svelte:head>

<h1>Types of Recurrent Neural Networks</h1>
<div class="separator"/>

<Container>
  <p>It is very common to categorize a recurrent neural network according to the types of inputs the network receives and the relevant outputs the network generates.</p>
  <p>The architecture that we have discussed in the last section is a so called sequence to sequence architecture, often abbrevieated as <Highlight>seq-to-seq</Highlight>. </p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 500 160">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="X {idx+1}" fontSize={12} x=50 y=150 width=25 height=15 color="var(--main-color-4)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color="var(--main-color-4)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>The architecture takes a series as an input and generates a series as an output. Each part of the output sequence is based solely on the previous inputs. The <Latex>Y_1</Latex> output is based on the <Latex>X_1</Latex> input, while <Latex>Y_2</Latex> is based on <Latex>X_1</Latex> and <Latex>X_2</Latex>. This architecture is very well suited for time series data. If we take temperature measurements as an example, we could try to forecast the temperature based on the history of the last n days. We mustn't use future data points for prediction, or we will introduce a problem that is known as <Highlight>data leakage</Highlight>. The first prediction might be based solely on the first measurement, while other predictions will incorporate more and more days.</p>
  
  <p>When the recurrent neural network takes a sequence as an input and generates a value (or a single vector) as an output, we are dealing with a <Highlight>seq-to-vector</Highlight> architecture.</p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 500 160">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="X {idx+1}" fontSize={12} x=50 y=150 width=25 height=15 color="var(--main-color-4)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color={idx===3 ? "var(--main-color-4)" : "var(--main-color-1)"}  />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
      <Border x=15 y=5 width=300 height=50 />
    </svg>
  </SvgContainer>
  <p>Practically an RNN produces outputs for each part of the sequence, but we choose to ignore non last outputs.</p>
  <p>The most common example for seq-to-vector is the so called <Highlight>sentiment analysis</Highlight>. The recurrent neural network might take a sentence as an input and generate a sentiment value, which could be a binary variable: 0 -> good sentiment, 1 -> bad sentiment.</p>

  <p>Obviously it is also possible to go the other way around. The neural network could take a single vector as an input and generate a sequence, hence this type of a network is called <Highlight>vec-to-seq</Highlight>.</p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 500 160">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          {#if idx === 0}
            <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
            <Block text="X {idx+1}" fontSize={12} x=50 y=150 width=25 height=15 color="var(--main-color-4)" />
          {/if}
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color={"var(--main-color-4)"}  />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>Image captioning is a great example for vet-to-seq. The network takes an image as an input and generates a text description.</p>
  <p>But what do we do if the length of the input sequence and the length of the output sequence might be inconsistent? Language translation is such a constellation. Different languages not only produce sentences of different lenghts to describe the same meaning, but the order of the words might also be completely different in the two languages.</p>
  <p>For that we have two use a two step recurrent neural network, basically combining a seq-to-vec and a vec-to-seq. In the first step we use a seq-to-vec to encode the whole meaning of the original sentence in a single vector. This part is called the <Highlight>encoder</Highlight>. In the second step, we use a vec-to-sec architecture to decode the encoded sentence into a sequence. This part is called the <Highlight>decoder</Highlight>. And the overall network is called <Highlight>encoder-decoder</Highlight>.</p>
  <SvgContainer maxWidth={"1000px"}>
    <svg viewBox="0 0 850 160">
      {#each Array(7) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          {#if idx < 3}
            <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
            <Block text="X {idx+1}" fontSize={12} x=50 y=150 width=25 height=15 color="var(--main-color-4)" />
          {/if}
          {#if idx >= 3}
            <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
            <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color="var(--main-color-4)" />
          {/if}
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
      <Block x=150 y=22 width=100 height=25 text="Encoder" color=none fontSize={20} />
      <Block x=600 y=140 width=100 height=25 text="Decoder" color=none fontSize={20} />
      <Border x=10 y=50 width=310 height=110 />
      <Border x=370 y=1 width=475 height=110 />
    </svg>
  </SvgContainer>
  <div class="separator" />
</Container>

