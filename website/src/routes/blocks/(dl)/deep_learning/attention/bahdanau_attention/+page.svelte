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
  import Plus from "$lib/diagram/Plus.svelte";
  import Border from "$lib/diagram/Border.svelte";
  const references = [
    {
        author: "Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio",
        title: "Neural machine translation by jointly learning to align and translate",
        journal: "2014",
    }
  ]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Bahdanau Attention</title>
  <meta
    name="description"
    content="Bahdanau attention builds on the classical encoder-decoder rnn architecture, but instead of taking a single hidden unit input, the decoder calculates attention scores and weights all output of the decoder."
  />
</svelte:head>

<h1>Bahdanau Attention</h1>
<div class="separator" />

<Container>
  <p>In order to understand modern attention architectures, it makes sense to study the historical context in which these architectures were developed and the problems that the new systems tried to solve. For that purpose let's reiterate the encoder-decoder architecture from the last chapter and try to figure out in what regard this design is problematic.</p>

  <SvgContainer maxWidth={"1000px"}>
    <svg viewBox="0 0 850 160">
      {#each Array(7) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          {#if idx < 3}
            <Arrow strokeWidth=2 data={[{x: 50, y:140}, {x:50, y: 100}]} />
            <Block text="X {idx+1}" fontSize={12} x=50 y=150 width=25 height=15 color="var(--main-color-4)" />
          {/if}
          {#if idx >= 3}
            <Arrow strokeWidth=2 data={[{x: 50, y:55}, {x:50, y: 10}]} />
            <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color="var(--main-color-4)" />
          {/if}
          <Arrow strokeWidth=2 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
      <Block x=150 y=22 width=150 height=35 text="Encoder" color=none fontSize={30} />
      <Block x=600 y=140 width=150 height=35 text="Decoder" color=none fontSize={30} />
      <Border x=10 y=50 width=310 height=110 />
      <Border x=370 y=1 width=475 height=110 />
    </svg>
  </SvgContainer>

  <p>Let's imagine we are trying to solve a translation task using the above architecture. The encoder takes the sequence in the original language as input and returns a single vector, marked as H_3 in the illustration above. In other words the whole meaning of the original language is compressed into a single vector. As this vector moves through the decoder it gets adjusted and the original meaning of the sentence gets more and more diluted. By the time this vector arrives at the later stages of the decoder hardly anything is left from the input language and the translation quality suffers.</p>
  <p>The so called Bahdanau attention was designed to tackle this problem. The research paper was released in the year 2014 paper and is called after its main author, Dzmitry Bahdanau <InternalLink id={1} type={"reference"}/>. The simple (yet powerful) ideat that the paper introduced, was to take all outputs from the encoder as inputs into the decoder at each step of the decoding process and thus reducing the information bottleneck, that results on relying on solely one vector.</p>
  <p>So instead on focusing on H_3 alone, in the below illustration the decoder has to use Y_1 - Y_3 as inputs.</p>

  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 400 250">
      {#each Array(3) as _, idx}
        <g transform="translate({10+idx*120 - 20}, 30)">
          <Arrow strokeWidth=2 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Block text="X {idx+1}" fontSize={12} x=50 y=155 width=25 height=15 color="var(--main-color-4)" />
          <Arrow strokeWidth=2 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
          <Arrow strokeWidth=2 data={[{x: 50, y:50}, {x:50, y: 10}]} />
          <Block text="H {idx+1}" fontSize={12} x=50 y=-10 width=25 height=15 color="var(--main-color-1)" />
        </g>
      {/each}
      <Block x=340 y=220 width=100 height=25 text="Encoder" color=none fontSize={20} />
    </svg>
  </SvgContainer>

  <p>Obviously not all encoder outputs are relevant for each part of the decoder section. So the decoder pays attention to certain parts of the encoder outputs and weighs each output accordingly. The weighted sum is eventually used as the input into the decoder network.</p>

  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 500 600">
      <!-- decoder -->
      {#each Array(4) as _, idx}
        <g transform="translate({50+idx*120 - 20}, 30)">
          <Arrow strokeWidth=2 dashed={true} strokeDashArray="4 4" data={[{x: 50, y:230}, {x:50, y: 100}]} />
          <Arrow strokeWidth=2 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text="S_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
          <Arrow strokeWidth=2 data={[{x: 50, y:50}, {x:50, y: 10}]} />
          <Block text="Y {idx+1}" fontSize={12} x=50 y=-10 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}

      <!-- encoder -->
      {#each Array(3) as _, idx}
        <g transform="translate({50+idx*120 - 20}, 390)">
          <Arrow strokeWidth=2 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Block text="X {idx+1}" fontSize={12} x=50 y=155 width=25 height=15 color="var(--main-color-4)" />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Arrow strokeWidth=2 data={[{x: 50, y:50}, {x:50, y: 10}]} />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
          {#if idx !== 2}
            <Arrow strokeWidth=2 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          {:else}
            <Arrow strokeWidth=2 data={[{x: 70, y:75}, {x:140, y: 75}, {x:140, y: 200}, {x:-250, y: 200}, {x:-250, y: -285}, {x:-220, y: -285}] } />
          {/if}
          <Block text="H {idx+1}" fontSize={12} x=50 y=-10 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}

      <!-- context connections from encoder -->
      {#each Array(4) as _, idxTop}
        <Plus x={100+idxTop*120-20} y={285} radius={10} offset={4} />
        {#each Array(3) as _, idxBot}
            <Arrow strokeWidth=2 dashed=True strokeDashArray="4 4" showMarker={false} data={[{x: 100 + idxBot*120-20, y: 365}, {x: 100+idxTop*120-20, y: 300}]} />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>One relevant question still remains is: "how do we calculate the attention weights?"</p>
  <p>In the very first step we calculate the so called energy, <Latex>{String.raw`e_{ij} = a(s_{i-1}, h_j)`}</Latex>  . The energy measures the the strenght of the connection between some output of the decoder <Latex>h_j</Latex> and the previous decoder output <Latex>{String.raw`s_{i-1}`}</Latex> for each decoding time step <Latex>i</Latex>. If we want to generate the S_2 state, we calculte the following energies: <Latex>{String.raw`e_{21} = a(s_{1}, h_1)`}</Latex>, <Latex>{String.raw`e_{22} = a(s_{1}, h_2)`}</Latex> and <Latex>{String.raw`e_{23} = a(s_{1}, h_3)`}</Latex>. The higher the energy, the higher the attention to that particular encoder output is going to be. The function <Latex>a</Latex> is implemented as a neural network, that is trained simultaneously with other parts of the whole encoder-decoder architecture.</p>
  <p>Calculating the actual attention weights <Latex>\alpha</Latex> for the decoder input <Latex>i</Latex> towards the encoder output <Latex>j</Latex> is just a matter of using the engergy as an input into the softmax function.</p>
  <Latex>{String.raw`\alpha_{ij} = \dfrac{\exp(e_{ij})}{\sum_j \exp(e_{ij})}`}</Latex> 
  <p>Finally we use the attention weights to calculate the weighted sum of encoder outputs called a <Highlight>context vector</Highlight>, which is used as the input into the decoder.</p>
  <Latex>{String.raw`c_i = \sum_j \alpha_{ij} h_j`}</Latex> 
  <p>You might be uncertain regarding some parts of the implementation of this particular architecture. Hopefully those details we become clearer, when we implement this architecture in the next section.</p>  
</Container>
<Footer {references} />
