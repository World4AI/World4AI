<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Alert from "$lib/Alert.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
</script>

<svelte:head>
  <title>Types of Recurrent Neural Networks - World4AI</title>
  <meta
    name="description"
    content="There are different types of recurrent neural network architectures. Architectures like seq-to-seq, seq-to-vec, vec-to-seq and encoder-decoder."
  />
</svelte:head>

<h1>Types of Recurrent Neural Networks</h1>
<div class="separator" />

<Container>
  <p>
    Recurrent neural networks are typically classified into different
    categories. This section is dedicated to that topic.
  </p>
  <p>
    The architecture that we have discussed in the last section is a so called
    sequence to sequence architecture, often abbrevieated as <Highlight
      >seq-to-seq</Highlight
    >.
  </p>
  <Alert type="info"
    >A seq-to-seq architecture takes a sequence as an input and generates a
    different sequence.</Alert
  >
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 470">
      {#each Array(4) as _, idx}
        <g transform="translate(0, {idx * 120 - 20})">
          <Arrow
            strokeWidth="2"
            data={[
              { x: 31, y: 45 },
              { x: 76, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 120, y: 45 },
              { x: 164, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 100, y: 62 },
              { x: 100, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Block x="100" y="45" width="30" height="30" class="fill-slate-500" />
          <Block
            text="x_{idx + 1}"
            type="latex"
            fontSize={12}
            x="15"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="y_{idx + 1}"
            fontSize={12}
            x="185"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    The network takes a series of vectors (or scalars) <Latex
      >{String.raw`\mathbf{x_n}`}</Latex
    >
    as input and generates the same number of outputs <Latex
      >{String.raw`\mathbf{y_n}`}</Latex
    > .
  </p>
  <p>This architecture is for example well suited for time series data.</p>
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 470">
      {#each Array(4) as _, idx}
        <g transform="translate(0, {idx * 120 - 20})">
          <Arrow
            strokeWidth="2"
            data={[
              { x: 31, y: 45 },
              { x: 76, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 120, y: 45 },
              { x: 164, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 100, y: 62 },
              { x: 100, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Block x="100" y="45" width="30" height="30" class="fill-slate-500" />
          <Block
            text="x_{idx + 1}"
            type="latex"
            fontSize={12}
            x="15"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="x_{idx + 2}"
            fontSize={12}
            x="185"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Let's say we would like to predict the temperature of the next day <Latex
      >{String.raw`x_{t+1}`}</Latex
    >based on the current temperature .<Latex>{String.raw`x_t`}</Latex>
    We would offset the time series by a day and use previous temperature and the
    hidden vector <Latex>{String.raw`\mathbf{h_n}`}</Latex> to predict some values.
  </p>
  <p>
    When the recurrent neural network takes a sequence as an input and generates
    a value (or a single vector) as an output, we are dealing with a <Highlight
      >seq-to-vec</Highlight
    > architecture.
  </p>
  <Alert type="info"
    >A seq-to-vec archtitecture takes a sequence as an input and generates a
    vector as an output.</Alert
  >
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 470">
      {#each Array(4) as _, idx}
        <g transform="translate(0, {idx * 120 - 20})">
          <Arrow
            strokeWidth="2"
            data={[
              { x: 31, y: 45 },
              { x: 76, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 120, y: 45 },
              { x: 164, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 100, y: 62 },
              { x: 100, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Block x="100" y="45" width="30" height="30" class="fill-slate-500" />
          <Block
            text="x_{idx + 1}"
            type="latex"
            fontSize={12}
            x="15"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="y_{idx + 1}"
            fontSize={12}
            x="185"
            y="45"
            width="25"
            height="25"
            class={idx === 3 ? `fill-blue-100` : `fill-red-500`}
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Practically an RNN produces outputs for each part of the sequence, but we
    choose to ignore non last outputs.
  </p>
  <p>
    The most common example for seq-to-vec is the so called <Highlight
      >sentiment analysis</Highlight
    >. The recurrent neural network might take a sentence as an input and
    generate a sentiment value, which could be a binary variable: 0 means good
    sentiment, 1 means bad sentiment.
  </p>

  <p>
    Obviously it is also possible to go the other way around. The neural network
    could take a single vector as an input and generate a sequence, hence this
    type of a network is called <Highlight>vec-to-seq</Highlight>.
  </p>
  <Alert type="info"
    >A vec-to-seq archtitecture takes a vector as an input and generates a
    sequence as an output.</Alert
  >
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 470">
      {#each Array(4) as _, idx}
        <g transform="translate(0, {idx * 120 - 20})">
          {#if idx === 0}
            <Arrow
              strokeWidth="2"
              data={[
                { x: 31, y: 45 },
                { x: 76, y: 45 },
              ]}
              dashed={true}
              moving={true}
              strokeDashArray="4 4"
            />
          {/if}
          <Arrow
            strokeWidth="2"
            data={[
              { x: 120, y: 45 },
              { x: 164, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 100, y: 62 },
              { x: 100, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Block x="100" y="45" width="30" height="30" class="fill-slate-500" />
          {#if idx === 0}
            <Block
              text="x"
              type="latex"
              fontSize={12}
              x="15"
              y="45"
              width="25"
              height="25"
              class="fill-blue-100"
            />
          {/if}
          <Block
            type="latex"
            text="y_{idx + 1}"
            fontSize={12}
            x="185"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Image captioning is a great example for vec-to-seq. The network takes an
    image as an input and generates a text description.
  </p>
  <p>
    But what do we do if the length of the input sequence and the length of the
    output sequence might be inconsistent? Language translation is such a
    constellation. Different languages not only produce sentences of different
    lenghts to describe the same meaning, but the order of the words might also
    be completely different in the two languages. In order to deal with such
    problems we need to construct an <Highlight>encoder-decoder</Highlight> architecture.
  </p>
  <Alert type="info"
    >The encoder-decoder architecture constructs a model from two distinct
    parts: the encoder and the decoder.</Alert
  >
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 170">
      <Arrow
        strokeWidth="2"
        data={[
          { x: 30, y: 25 },
          { x: 100, y: 25 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="2"
        data={[
          { x: 90, y: 145 },
          { x: 164, y: 145 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="2"
        data={[
          { x: 155, y: 40 },
          { x: 155, y: 85 },
          { x: 45, y: 85 },
          { x: 45, y: 115 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Block
        x="155"
        y="25"
        width="85"
        height="40"
        text="Encoder"
        fontSize="18"
        class="fill-blue-200"
      />
      <Block
        x="45"
        y="145"
        width="85"
        height="40"
        text="Decoder"
        fontSize="18"
        class="fill-lime-200"
      />
      <Block
        type="latex"
        fontSize={12}
        x="15"
        y="25"
        width="25"
        height="25"
        class="fill-blue-300"
      />
      <Block
        type="latex"
        fontSize={12}
        x="185"
        y="145"
        width="25"
        height="25"
        class="fill-lime-300"
      />
    </svg>
  </SvgContainer>
  <p>
    In the first step we use a seq-to-vec to encode the whole meaning of the
    original sentence in a single vector. This part is called the <Highlight
      >encoder</Highlight
    >.
  </p>
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 470">
      {#each Array(4) as _, idx}
        <g transform="translate(0, {idx * 120 - 20})">
          <Arrow
            strokeWidth="2"
            data={[
              { x: 31, y: 45 },
              { x: 76, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 100, y: 62 },
              { x: 100, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Block x="100" y="45" width="30" height="30" class="fill-slate-500" />
          <Block
            text="x_{idx + 1}"
            type="latex"
            fontSize={12}
            x="15"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    In the second step, we use a vec-to-sec architecture to decode the encoded
    sentence into a sequence. This part is called the <Highlight
      >decoder</Highlight
    >. The decoder receives the last hidden value of the encoder, which is
    supposed to contain the encoded meaning of the original language. This
    vector is used as the initial hidden value for the decoder.
  </p>
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 250">
      {#each Array(2) as _, idx}
        <g transform="translate(0, {idx * 120 - 20})">
          <Arrow
            strokeWidth="2"
            data={[
              { x: 120, y: 45 },
              { x: 164, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          {#if idx === 0}
            <Arrow
              strokeWidth="2"
              data={[
                { x: 100, y: 0 },
                { x: 100, y: 140 },
              ]}
              dashed={true}
              moving={true}
              strokeDashArray="4 4"
            />
          {:else}
            <Arrow
              strokeWidth="2"
              data={[
                { x: 100, y: 62 },
                { x: 100, y: 140 },
              ]}
              dashed={true}
              moving={true}
              strokeDashArray="4 4"
            />
          {/if}
          <Block x="100" y="45" width="30" height="30" class="fill-slate-500" />
          <Block
            type="latex"
            text="y_{idx + 1}"
            fontSize={12}
            x="185"
            y="45"
            width="25"
            height="25"
            class="fill-lime-200"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-orange-200"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Often the decoder and the encoder are actually two distinct RNNs with
    different weights, which can facilitate training.
  </p>
  <p>
    Before we move on to the next section let us mention, that there are
    different implementation details of very similar implementations. The
    definitios above are just rough ideas and should not be regarded as a holy
    grail.
  </p>
  <Alert type="warning">
    There are different flavours of the same architectures out there, so when
    you encounter a certain definition in an academic paper or a blog post, make
    sure you understand the exact intentions of the author.
  </Alert>
  <div class="separator" />
</Container>
