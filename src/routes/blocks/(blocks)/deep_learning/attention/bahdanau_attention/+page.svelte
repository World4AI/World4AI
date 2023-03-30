<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  const references = [
    {
      author: "Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio",
      title:
        "Neural machine translation by jointly learning to align and translate",
      journal: "2014",
    },
  ];
</script>

<svelte:head>
  <title>Bahdanau Attention - World4AI</title>
  <meta
    name="description"
    content="Bahdanau attention builds on the classical encoder-decoder RNN architecture, but instead of taking a single hidden unit input, the decoder calculates attention scores and weights all output of the decoder, thus increasing the performance of the neural network."
  />
</svelte:head>

<h1>Bahdanau Attention</h1>
<div class="separator" />

<Container>
  <p>
    In order to understand modern attention architectures, it makes sense to
    study the historical context in which these architectures were developed and
    the problems that the new systems tried to solve. For that purpose let's
    remember the encoder-decoder architecture from the last chapter and try to
    figure out in what regard this design might be problematic.
  </p>
  <p>
    Let's imagine we are trying to solve a translation task with and
    enocder-decoder architecture. The encoder takes the sequence in the original
    language as input and returns a single vector, marked as <Latex>h_4</Latex>.
    In other words the whole meaning of the original language is compressed into
    a single vector.
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
    The decoder uses this hidden vector and previously generated words as input
    and generates a translation one word at a time. As the hidden vector moves
    through the decoder it gets modified and the original meaning of the
    sentence gets more and more diluted. By the time this vector arrives at the
    end of the decoder, hardly anything is left from the input language and the
    translation quality suffers.
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
            text="y_{idx}"
            type="latex"
            fontSize={12}
            x="15"
            y="45"
            width="25"
            height="25"
            class="fill-lime-200"
          />
          <Block
            type="latex"
            text="s_{idx + 1}"
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
    In order to tackle the above problem Dzmitry Bahdanau and his colleagues
    developed the so called <Highlight>Bahdanau Attention</Highlight
    ><InternalLink id={1} type={"reference"} />. The authors had a simple yet
    powerful idea to take all outputs from the encoder as inputs into the
    decoder at each step of the decoding process and thus reduced the
    information bottleneck that results from relying on solely one vector.
    Obviously not all encoder outputs are relevant equally for each part of the
    decoder section. So at each step of the decoding (translation) process the
    decoder pays attention to certain parts of the encoder outputs and weighs
    each accordingly. The weighted sum is eventually used as the input into the
    decoder network. In this section we will refer to this variable as the
    context <Latex>c_i</Latex>.
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
          <Arrow
            strokeWidth="2"
            data={[
              { x: 100, y: 100 },
              { x: 180, y: 250 - idx * 115 },
            ]}
            dashed={true}
            moving={true}
            showMarker={false}
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
      <Block
        x="180"
        y="240"
        width="30"
        height="30"
        class="fill-yellow-200"
        type="latex"
        text={"c_i"}
        fontSize={15}
      />
    </svg>
  </SvgContainer>
  <p>
    There might be different strategies to use the context as an input to the
    decoder. In our implementation we simply concatenate the previous decoder
    output with the context and use that as the input.
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
            text="y_{idx}"
            type="latex"
            fontSize={12}
            x="15"
            y="45"
            width="25"
            height="25"
            class="fill-lime-200"
          />
          <Block
            text="c_{idx + 1}"
            type="latex"
            fontSize={12}
            x="15"
            y="75"
            width="25"
            height="25"
            class="fill-yellow-200"
          />
          <Block
            type="latex"
            text="s_{idx + 1}"
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
    In order to calculate the context we have to take a series of steps. In the
    very first step we calculate the so called energy, <Latex
      >{String.raw`e_{ij} = a(s_{i-1}, h_j)`}</Latex
    > . At each decoding/translation step we measure the engergy between the previously
    generated hidden state of the decoder <Latex>{String.raw`s_{i-1}`}</Latex> and
    each of the enocder outputs. The energy measures the the strength of the connection
    between a decoder output <Latex>h_j</Latex> and the previous decoder output <Latex
      >{String.raw`s_{i-1}`}</Latex
    >. Given that we have 4 encoder outputs if we want to generate the energies
    needed for the
    <Latex>s_2</Latex> state, we calculte the following energies: <Latex
      >{String.raw`e_{21} = a(s_{1}, h_1)`}</Latex
    >, <Latex>{String.raw`e_{22} = a(s_{1}, h_2)`}</Latex>, <Latex
      >{String.raw`e_{23} = a(s_{1}, h_3)`}</Latex
    > and <Latex>{String.raw`e_{23} = a(s_{1}, h_4)`}</Latex> . The higher the energy,
    the higher the attention to that particular encoder output is going to be. The
    function <Latex>a</Latex> is implemented as a neural network, that is trained
    jointly with other parts of the whole encoder-decoder architecture. Calculating
    the actual attention weights <Latex>\alpha</Latex> for the decoder input <Latex
      >i</Latex
    > towards the encoder output <Latex>j</Latex> is just a matter of using the engergies
    as an input into the softmax function.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\alpha_{ij} = \dfrac{\exp(e_{ij})}{\sum_j \exp(e_{ij})}`}</Latex
    >
  </div>
  <p>
    Finally we use the attention weights to calculate the weighted sum of
    encoder outputs, the <Highlight>context vector</Highlight>, which is used as
    the input into the decoder together with the previously generated word
    token.
  </p>
  <div class="flex justify-center">
    <Latex>{String.raw`c_i = \sum_j \alpha_{ij} h_j`}</Latex>
  </div>
</Container>
<Footer {references} />
