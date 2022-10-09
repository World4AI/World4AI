<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Border from "$lib/diagram/Border.svelte";

  const references = [
    {
        author: "van den Oord, Aaron and Vinyals, Oriol and Kavukcuoglu, Koray",
        title: "Neural Discrete Representation Learning",
        journal: "Advances in Neural Information Processing Systems",
        year: "2017",
        pages: "",
        volume: "30",
        issue: "",
    }
  ]

</script>

<svelte:head>
  <title>World4AI | Deep Learning | VQ-VAE</title>
  <meta
    name="description"
    content="A vector quantazed variational autoencoder (VQ-VAE) produces discrete encoder output values. A codebook with embedding values is stored and the encoder output measures the distance between those outputs and embeddings. The corresponding indices are the discrete quantised values."
  />
</svelte:head>

<h1>VQ-VAE</h1>
<div class="separator"></div>

<Container>
  <p>The vector quantised variational autoencoder <InternalLink type="reference" id={1} />, <Highlight>VQ-VAE</Highlight>, was developed by DeepMind in 2017. While in deep learning terms this paper is considered to be quite old, the techniques that we learn in this section are still relevant to this day. Modern text to image generative models rely heavily on VQ-VAE.</p>
  <p>The major idea of the paper is to quantise encoder outputs, which means that the latent variables produced by the encoder are transformed into discrete values. For that purpose we require a so called codebook.</p>  

  <SvgContainer maxWidth={"400px"}>
    <svg viewBox = "0 0 250 170">
      <Block x=125 y=15 width=110 height=25 text="CODEBOOK" fontSize=15 color="var(--main-color-1)"/>
      {#each Array(7) as _, idx}
        <Block x={25 + idx*25} y={100} width=20 height=120 color="var(--main-color-2)" text="e_{idx+1}" fontSize=10 type="latex" />
      {/each}
      <Block x={25 + 7*25} y={100} width=20 height=120 color="var(--main-color-2)" text="..." fontSize=10 type="latex" />
      <Block x={25 + 8*25} y={100} width=20 height=120 color="var(--main-color-2)" text="e_K" fontSize=10 type="latex" />
      <Border x={5} y={35} width={240} height={130} />
    <svg>
  </SvgContainer>
  <p>The codebook contains <Latex>{String.raw`K`}</Latex> embeddings of dimensionality <Latex>{String.raw`D`}</Latex>. Encoder outputs <Latex>{String.raw`z_e(x)`}</Latex> are compared to each individual embedding and the index <Latex>{String.raw`k`}</Latex> of the nearest neighbour is used as the quantised value:<Latex>{String.raw`k = \arg\min_j ||z_e(x) - e_j ||`}</Latex>. Let's look at a stylized example of a convolutional VQ-VAE and try to understand what exactly that means.</p>

  <SvgContainer maxWidth={"1000px"}>
    <svg viewBox = "0 0 800 300">
      <!-- codebook -->
      <Block x=385 y=20 width=110 height=25 text="CODEBOOK" fontSize=15 color="var(--main-color-1)"/>
      {#each Array(6) as _, colIdx}
        <Block x={320 + colIdx*25} y={50} width=20 height=20 text={colIdx+1} color="none" fontSize=12 />
        {#each Array(3) as _, rowIdx}
          <Block x={320 + colIdx*25} y={80+ rowIdx * 25} width=20 height=20 color="var(--main-color-2)" fontSize=10 type="latex" />
        {/each}
      {/each}

      <!-- encoder outputs -->
      <Block x=80 y=130 width=140 height=25 text="Encoder Output" fontSize=15 color="var(--main-color-1)"/>
      {#each Array(5) as _, rowIdx}
        {#each Array(5) as _, colIdx}
          {#each Array(3) as _, channelIdx}
            <Block x={45-channelIdx*10 + colIdx*25} y={180 - channelIdx*10 + rowIdx * 25} width={20} height={20} color={colIdx===4 && rowIdx===0 ? "var(--main-color-1)" : "var(--main-color-3)"} />
          {/each}
        {/each}
      {/each}

      <!-- decoder inputs -->
      <Block x=710 y=130 width=140 height=25 text="Decoder Output" fontSize=15 color="var(--main-color-1)"/>
      <g transform="translate(630, 0)">
        {#each Array(5) as _, rowIdx}
          {#each Array(5) as _, colIdx}
            {#each Array(3) as _, channelIdx}
              <Block x={45-channelIdx*10 + rowIdx*25} y={180 - channelIdx*10 + colIdx * 25} width={20} height={20} color="var(--main-color-4)" />
            {/each}
          {/each}
        {/each}
      </g>

      <!-- quantised outputs -->
      {#each Array(5) as _, rowIdx}
        {#each Array(5) as _, colIdx}
         <Block x={335 + rowIdx*25} y={180 + colIdx * 25} width={20} height={20} color="var(--main-color-2)" text={Math.ceil(Math.random() * 6)} fontSize=15/>
        {/each}
      {/each}
    <svg>
  </SvgContainer>
  <p>Let's assume for simplicity, that the encoder produces a 5x5x3 image. The number of channels has to match the embedding dimensionality<Latex>{String.raw`K`}</Latex>, because we essentially compare each of the embeddings from the codebook to each of the vectors along the channel dimension. So in our case the codebook has vectors of size 3. For each pixel location, the quantised (blue) matrix contains the index of the embedding, that is closest to the encoder output at that particular pixel location. The decoder takes that quantised matrix and uses the corresponding embeddings from the codebooks as input, instead of the encoder output.</p>
  <p>This lookup approach poses a problem, as we can not use backpropagation from the decoder input to the encoder output. Instead we assume that the gradients of the decoder input and the decoder output are similar and simply copy them from the decoder to the encoder.</p>
  <p>The loss function consists of three parts. First we measure the usual reconstruction loss between the intput<Latex>{String.raw`x`}</Latex> and the decoder output <Latex>{String.raw`z_q(x)`}</Latex>. Second we measure the VQ loss, as the difference between the frozen encoder outputs and the embeddings. Basically we try to move embeddings closer to the encoder outputs. Lastly we measure the commitment loss as the difference between frozen embeddings and encoder outputs. That makes sure that the encoder commits to a particular embedding and does not grow out of proportions.</p>
  <Latex>{String.raw`L = \underbrace{||x - z_q(x)||^2_2}_{\text{Reconstruction Loss}} + \underbrace{||sg[z_e(x)] - e||^2_2}_{\text{VQ Loss}} + \beta\underbrace{||z_e(x) - sg(e)||^2_2}_{\text{Commitment Loss}}`}</Latex> 
  <div class="separator"></div>
  <h2>Autoregressive Prior - PixelCNN</h2>
  <div class="separator"></div>
  <h2>VQ-VAE2</h2>
  <div class="separator"></div>
</Container>
<Footer {references} />
