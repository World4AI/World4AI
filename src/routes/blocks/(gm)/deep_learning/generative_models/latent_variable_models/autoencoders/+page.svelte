<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte"; import Border from "$lib/diagram/Border.svelte";

  const svgOffset = 40;
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Autoencoders </title>
  <meta
    name="description"
    content="An autoencoder uses an encoder to map its input to latent space and uses a decoder to transform the latent variable back into the original image. The bottleneck is usually a lower dimensional vector that the input space, therefore an autoencoder compresses the input information."
  />
</svelte:head>

<h1>Autoencoders</h1>
<div class="separator" />

<Container>
  <p>An <Highlight>autoencoder</Highlight> is a neural network architecture, that maps an input into a lower dimensional space and reconstructs the original input from the lower space.</p>
  <p>We use some variable <Latex>{String.raw`\mathbf{X}`}</Latex> as an input into an autoencoder. This could be a vector input in a fully connected neural network, but an autoencoder is also often used with images in combination with a convolutional neural network, therefore we could be dealing with a matrix. An autoencoder is trained in an unsupervised way, without using any additional labels, specifically the input and the training labels of an autoencoder are identical. So if we use an image as an input, we expect the output of the neural network, <Latex>{String.raw`\mathbf{X}'`}</Latex>, to be as close as possible to that input image.</p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox = "0 0 400 250">
      <!-- Arrows -->
      <Arrow data={[{x: svgOffset+12.5+5, y: 90-75}, {x: svgOffset+80*2-12.5-10, y: 90-20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+12.5 +5, y: 90+75}, {x: svgOffset+80*2-12.5-10, y: 90+20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50} />

      <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90-20}, {x: svgOffset+80*4-12.5-10, y: 90-75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90+20}, {x: svgOffset+80*4-12.5-10, y: 90+75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>

      <!-- Encoder -->
      <Block x={svgOffset+0} y={90} width={25} height={150} text={String.raw`\mathbf{X}`} type="latex" fontSize="20" color="var(--main-color-3)"/>
      <Block x={svgOffset+80} y={90} width={25} height={100} color="var(--main-color-2)"/>
      <!-- Bottleneck -->
      <Block x={svgOffset+80*2} y={90} width={25} height={40} text={String.raw`\mathbf{z}`} type="latex" fontSize="20" color="var(--main-color-1)" />
      <!-- Decoder -->
      <Block x={svgOffset+80*3} y={90} width={25} height={100} color="var(--main-color-2)" />
      <Block x={svgOffset+80*4} y={90} width={25} height={150} text={String.raw`\mathbf{X'}`} type="latex" fontSize="20" color="var(--main-color-3)" />
        
      <!-- Borders -->
      <Border x={10} y={5} width={140} height={180}/>
      <Border x={250} y={5} width={140} height={180}/>

      <!-- Labels -->
      <Block x={50} y={230} width={80} height={25} text="Encoder" fontSize="15"/>
      <Block x={200} y={40} width={80} height={20} text="Bottleneck" fontSize="12"/>
      <Block x={350} y={230} width={80} height={25} text="Decoder" fontSize="15"/>
    <svg>
  </SvgContainer>

  <p>The neural network consists of two components: an <Highlight>encoder</Highlight> and a <Highlight>decoder</Highlight>. </p>
  <p>The encoder takes the input image <Latex>{String.raw`\mathbf{X}`}</Latex> and produces the latent variable vector <Latex>{String.raw`\mathbf{z}`}</Latex>. The dimensionality of hidden values keeps decreasing, until arrive at so called bottleneck. By decreasing the dimensionality, we compress the information, that is contained in the input, until we reach the highest compression point with the latent variable. </p>
  <p>The decoder takes the latent variable vector <Latex>{String.raw`\mathbf{z}`}</Latex> as an input and tries to uncompress the information into the original space. The neural network has to learn to produce <Latex>{String.raw`\mathbf{X'}`}</Latex>, that is as close as possible to the input. As we squeeze a high dimensional image into a low dimensional vector, the compression is lossy and the output image might lose some detail, but if the network is expressive enough, the loss won't be dramatic.</p>
  <p>Intuitively we could argue, that the neural network removes all the unnecessary noise, until only the relevant characteristics of the data that are contained in the latent space are left.</p>
  <p>In our notebook we will use this architecture to actually compress the images into a lower dimensional space. Yet we would like to use an autoencoder in order to generate new images and we can utilize the decoder for that purpose. Think about it, the decoder takes the latent variable vector and produces an image out of it. The problem with this vanilla autoencoder is that we have no built-in mechanism to sample new latent variables from which we could generate the images. A variational autoencoder on the other hand is much better suited as a generative model. We will study and utilze this architecture in the next section.</p>
  <div class="separator"></div>
</Container>
