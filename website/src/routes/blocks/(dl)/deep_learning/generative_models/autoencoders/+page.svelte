<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte"; import Border from "$lib/diagram/Border.svelte";

  const notes = ["The word latent comes from the latin word 'lateo', which means 'to be hidden'. You can think about a latent variable as you usually think about hidden features, but you will often find, that especially with generative models, the word 'latent' is used very often."];
  const svgOffset = 40;

  const numberOne = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ]

  const numberTwo = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
  ]

  const numberSix = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
  ]
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

  <p>We use some variable <Latex>{String.raw`X`}</Latex> as an input into an autoencoder. This could be a vector input in a fully connected neural network, but an autoencoder is also often used in combination with a convolutional neural network, therefore we could be dealing with a matrix. An autoencoder is trained in an unsupervised way, without using any additional labels, specifically the input and the training labels of an autoencoder are identical. So if we use an image as an input, we expect the output of the neural network, <Latex>{String.raw`X'`}</Latex>, to be as close as possible to that input image. When you encounter an autoencoder for the first time, you might think that this task is useless, but this is not the case.</p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox = "0 0 400 250">
      <!-- Arrows -->
      <Arrow data={[{x: svgOffset+12.5+5, y: 90-75}, {x: svgOffset+80*2-12.5-10, y: 90-20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+12.5 +5, y: 90+75}, {x: svgOffset+80*2-12.5-10, y: 90+20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50} />

      <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90-20}, {x: svgOffset+80*4-12.5-10, y: 90-75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90+20}, {x: svgOffset+80*4-12.5-10, y: 90+75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>

      <!-- Encoder -->
      <Block x={svgOffset+0} y={90} width={25} height={150} text="X" fontSize="20" color="var(--main-color-3)"/>
      <Block x={svgOffset+80} y={90} width={25} height={100} color="var(--main-color-2)"/>
      <!-- Bottleneck -->
      <Block x={svgOffset+80*2} y={90} width={25} height={40} text="Z" fontSize="20" color="var(--main-color-1)" />
      <!-- Decoder -->
      <Block x={svgOffset+80*3} y={90} width={25} height={100} color="var(--main-color-2)" />
      <Block x={svgOffset+80*4} y={90} width={25} height={150} text="X'" fontSize="20" color="var(--main-color-3)" />
        
      <!-- Borders -->
      <Border x={10} y={5} width={140} height={180}/>
      <Border x={250} y={5} width={140} height={180}/>

      <!-- Labels -->
      <Block x={50} y={230} width={80} height={25} text="Encoder" fontSize="15"/>
      <Block x={200} y={40} width={80} height={20} text="Bottleneck" fontSize="12"/>
      <Block x={350} y={230} width={80} height={25} text="Decoder" fontSize="15"/>
    <svg>
  </SvgContainer>

  <p>The neural network consists of two parts: an <Highlight>encoder</Highlight> and a <Highlight>decoder</Highlight>. The encoder takes the input <Latex>{String.raw`X`}</Latex> and produces a so called <Highlight>latent variable</Highlight><InternalLink type="note" id={1} />, <Latex>{String.raw`Z`}</Latex>. You should notice that the dimensionality of hidden values keeps decreasing, until we reach the so called bottleneck. By decreasing the dimensionality, we compress the information, that is contained in the input. The decoder takes the latent variable <Latex>{String.raw`Z`} </Latex> as an input and tries to uncompress the information into the original space. The neural network has to learn to produce <Latex>{String.raw`X'`}</Latex>, that is as close as possible to the input. As the compression is lossy, the output image might lose some information, but if the network is expressive enought, the loss won't be dramatic. </p>
  <p>This architecture can actually be used as a compression technique, but we would like to use an autoencoder in order to generate new images and we can utilize the decoder for that purpose. Think about it, the decoder takes the latent variable vector <Latex>Z</Latex> and produces an image out of it (if we are dealing with an image dataset). So technically we should be able to provide a different vector in order to generate a different value. But how do you pick the latent variable that produces the value that corresponds to your expectation? Let's say you trained an autoencoder on the MNIST dataset and would like to generate an image that contains the number 2. We could for example take all training samples that contain the number two, calculate the latent variable for each individual "2" and build a simple average. When we use the average as the input into our decoder, we should produce the "average" handwritten 2. When we deviate slightly from the average we should still produce the number two, but we shouldn't stray away too far from the center.</p>
  <p>But why does this work? Why can we build an average of latent variables? Let's look at a stylized example of digits and try to figure out what a latent variable actually is. </p>

  <SvgContainer maxWidth={"450px"}>
    <svg viewBox = "0 0 210 93">
      {#each numberOne as row, rowIdx}
        {#each row as cell, colIdx}
          <Block x={8+colIdx*13} y={8+rowIdx*13} width={10} height={10} color={cell === 1 ? 'black' : "none"} />
        {/each}
      {/each}
      <g transform="translate(70, 0)">
        {#each numberTwo as row, rowIdx}
          {#each row as cell, colIdx}
            <Block x={8+colIdx*13} y={8+rowIdx*13} width={10} height={10} color={cell === 1 ? 'black' : "none"} />
          {/each}
        {/each}
      </g>
      <g transform="translate(140, 0)">
        {#each numberSix as row, rowIdx}
          {#each row as cell, colIdx}
            <Block x={8+colIdx*13} y={8+rowIdx*13} width={10} height={10} color={cell === 1 ? 'black' : "none"} />
          {/each}
        {/each}
      </g>
    <svg>
  </SvgContainer>
  <p>The three numbers above have some similarities and some differences. All of them have a certain amount of lines. The number 1 consists of a single line, while you can detect 5 lines in the number 2 and 6. To differentiate between the number 2 and the number 6, we could for example notice that the number six contains has a closed form at the bottom, while the number two has a missing pixel at the bottom. So how could we transform that into a latent variable? Let's imagine that the first value in the latent vector represents whether the number is made of more than one line or not and the second variable is 1, if the number contains a closed box. Given that the three numbers would produce the following latent variables.</p>
  <Latex>{String.raw`
    \begin{bmatrix}
      0 \\
      0 \\
    \end{bmatrix}
    \begin{bmatrix}
      1 \\
      0 \\
    \end{bmatrix}
    \begin{bmatrix}
      1 \\
      1 \\
    \end{bmatrix}
  `}</Latex>

  <p>This is obviously a silly example and it should be clear, that actual latent variables have many more dimensions and are much smoother, but the following general idea holds. </p>
  <Alert type="info">
    Latent variables are encoded characteristics that are contained in the data.
  </Alert>
  <p>So when we are building an average latent variable for the number two, we create a number with average characteristics and the decoder knows how to map those characteristics to an actual image.</p>
  <p>While it is useful to cover this simple autoencoder architecture in order to understand more complex autoencoder networks, we have to mention that it is generally a bad idea to use this architecture for image generation. A variational autoencoder on the other hand is much better suited as a generative model. We will study and utilze this architecture in the next section.</p>
</Container>
<Footer {notes} />

<style>
  svg {
      border: 1px solid black;
    }
</style>
