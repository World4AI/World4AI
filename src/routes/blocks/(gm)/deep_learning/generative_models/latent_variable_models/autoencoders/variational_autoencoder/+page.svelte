<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";

  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte"; import Border from "$lib/diagram/Border.svelte";
  
  const references = [
    {
        author: "Diederik P. Kingma, Max Welling",
        title: "Auto-Encoding Variational Bayes",
        year: "2013",
    }
  ]

  const svgOffset = 40;
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Variational Autoencoder </title>
  <meta
    name="description"
    content="Unlike a regular autoencoder, a variational autoencoder (VAE) maps an input image to a latent variable distribution. We can use that distribution to sample new images."
  />
</svelte:head>

<h1>Variational Autoencoders</h1>
<div class="separator" />

<Container>
  <p>Variational autoencoders (VAE) are a family of autoencoders that are based on variational Bayesian methods. The mathematical background knowledge that is required to fully understand all intricacies of VAEs is quite extensive and can be intimidating for beginners. Therefore we will introduce VAEs using an intuitive setting first. In a second step we will delve deeper into the theoretical justification of variational autoencoders and provide additional sources for you to study. Fortunately the intuitive introduction is sufficient to implement a VAE from scratch in PyTorch, so you might skip the theoretical part for now and return to it at a later point, once you gained a better mathematical foundation.</p>
  <div class="separator" />

  <h2>VAE Intuition</h2>
  <p>The autoencoder, that we have discussed in the previous section follow a very simple approach: map an image to a latent vector <Latex>{String.raw`\mathbf{z}`}</Latex> and reconstruct the original image from from the latent variable.</p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox = "0 0 400 250">
      <!-- Arrows -->
      <Arrow data={[{x: svgOffset+12.5+5, y: 90-75}, {x: svgOffset+80*2-12.5-10, y: 90-20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+12.5 +5, y: 90+75}, {x: svgOffset+80*2-12.5-10, y: 90+20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50} />

      <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90-20}, {x: svgOffset+80*4-12.5-10, y: 90-75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90+20}, {x: svgOffset+80*4-12.5-10, y: 90+75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>

      <!-- Encoder -->
      <Block x={svgOffset+0} y={90} width={25} height={150} text="X" type="latex" fontSize="20" color="var(--main-color-3)"/>
      <Block x={svgOffset+80} y={90} width={25} height={100} color="var(--main-color-2)"/>
      <!-- Bottleneck -->
      <Block x={svgOffset+80*2} y={90} width={25} height={40} text={String.raw`\mathbf{z}`} type="latex" fontSize="20" color="var(--main-color-1)" />
      <!-- Decoder -->
      <Block x={svgOffset+80*3} y={90} width={25} height={100} color="var(--main-color-2)" />
      <Block x={svgOffset+80*4} y={90} width={25} height={150} text="X'" type="latex" fontSize="20" color="var(--main-color-3)" />
        
      <!-- Borders -->
      <Border x={10} y={5} width={140} height={180}/>
      <Border x={250} y={5} width={140} height={180}/>

      <!-- Labels -->
      <Block x={50} y={230} width={80} height={25} text="Encoder" fontSize="15"/>
      <Block x={200} y={40} width={80} height={20} text="Bottleneck" fontSize="12"/>
      <Block x={350} y={230} width={80} height={25} text="Decoder" fontSize="15"/>
    <svg>
  </SvgContainer>
  <p>While this procedure might be good enough for simple compression tasks, when it comes to generating new images, this simple mapping approach might break down. The latent space that is learned might not be smooth or continuous, so when you sample a latent vector that deviates slightly from the samples that the decoder saw during training, you will end up with a nonsence image.</p>
  <p>Variational autoencoders <InternalLink id={1} type="reference"/> on the other hand use a different approach. Instead of mapping the input image to a constant latent vector <Latex>{String.raw`\mathbf{z}`}</Latex>, they map the input <Latex>{String.raw`X`}</Latex> to a probability distribution of the latent variable.</p>
  <SvgContainer maxWidth={"700px"}>
    <svg viewBox = "0 0 470 250">
      <!-- Encoder Arrows -->
      <Arrow data={[{x: svgOffset+12.5+5, y: 90-75}, {x: svgOffset+80*2-12.5-10, y: 90-20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+12.5 +5, y: 90+75}, {x: svgOffset+80*2-12.5-10, y: 90+20}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50} />

      <!-- sampling arrows -->
      <Arrow data={[{x: svgOffset+80*2+12.5, y: 60}, {x: svgOffset+80*2+60, y: 80}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+80*2+12.5, y: 120}, {x: svgOffset+80*2+60, y: 100}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
      <Arrow data={[{x: svgOffset+80*2+12.5, y: 225}, {x: svgOffset+80*2+80, y: 225}, {x: svgOffset+80*2+80, y: 115}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>

      <!-- Encoder -->
      <Block x={svgOffset+0} y={90} width={25} height={150} text="X" type="latex" fontSize="20" color="var(--main-color-3)"/>
      <Block x={svgOffset+80} y={90} width={25} height={100} color="var(--main-color-2)"/>
      
      <!-- mu, sigma and epsilon -->
      <Block x={svgOffset+80*2} y={60} width={25} height={30} text="\boldsymbol \mu" type="latex" fontSize="20" color="var(--main-color-1)" />
      <Block x={svgOffset+80*2} y={120} width={25} height={30} text="\boldsymbol \sigma" type="latex" fontSize="20" color="var(--main-color-1)" />
      <Block x={svgOffset+80*2} y={225} width={25} height={30} text="\boldsymbol \epsilon" type="latex" fontSize="20" color="var(--main-color-1)" />

      <g transform="translate(80, 0)">
          <!-- Decoder Arrows -->
          <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90-20}, {x: svgOffset+80*4-12.5-10, y: 90-75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>
          <Arrow data={[{x: svgOffset+80*2+12.5+5, y: 90+20}, {x: svgOffset+80*4-12.5-10, y: 90+75}]} dashed={true} strokeDashArray="4 4" strokeWidth={1.5} moving={true} speed={50}/>

          <!-- Bottleneck -->
          <Block x={svgOffset+80*2} y={90} width={25} height={30} text={String.raw`\mathbf{z}`} type="latex" fontSize="20" color="var(--main-color-1)" />
    
          <!-- Decoder -->
          <Block x={svgOffset+80*3} y={90} width={25} height={100} color="var(--main-color-2)" />
          <Block x={svgOffset+80*4} y={90} width={25} height={150} text="X'" type="latex" fontSize="20" color="var(--main-color-3)" />
      </g>
      <!-- Borders -->
      <Border x={10} y={5} width={140} height={180}/>
      <Border x={300} y={5} width={160} height={180}/>
    
      <!-- Labels -->
      <Block x={50} y={230} width={80} height={25} text="Encoder" fontSize="15"/>
      <Block x={420} y={230} width={80} height={25} text="Decoder" fontSize="15"/>
    <svg>
  </SvgContainer>
  <p>Instead of producing the latent vector <Latex>{String.raw`\mathbf{x}`}</Latex> directly, the encoder generates two vectors: one containing the mean vector <Latex>{String.raw`\boldsymbol \mu`}</Latex> and the other containing the standard deviation vector <Latex>{String.raw`\boldsymbol \sigma`}</Latex>. The <Latex>{String.raw`\epsilon`}</Latex> vector is drawn from a standard normal distribution <Latex>{String.raw`\epsilon_i \sim \mathcal{N}(0, 1)`}</Latex>. The random vector is not part of the computational graph and is treated as a constant during the backpropagation step.</p>
  <p>Once the encoder has produced the relevant mean and standard deviation vectors, we can create the latent vector <Latex>{String.raw`\mathbf{z}`}</Latex>.</p>
  <Latex>{String.raw`\mathbf{z} = \boldsymbol \mu + \boldsymbol \sigma \odot \boldsymbol \epsilon \text{, where } \odot \text{ is elementwise multiplication}`}</Latex>
  <p>The last remaining puzzle is the loss function that we use to train a VAE, which consists of two parts: the reconstruction loss and a regularizer term.</p>
  <Latex>{String.raw`\mathcal{L} = Reconstruction + Regularizer`}</Latex>
  <p>The reconstruction loss we are going to use is the mean squared error between each pixel of the original image <Latex>{String.raw`X`}</Latex> and the reconstructed image <Latex>{String.raw`X'`}</Latex>.</p>
  <p>The regularizer on the other hand tries to make the distribution of the latent variables <Latex>{String.raw`\mathbf{z}`}</Latex> close to a normal distribution, by minimizing the following expression: <Latex>{String.raw`- \dfrac{1}{2} \sum^n_i (1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2)`}</Latex>, where n is the size of the latent variable vector.</p>
  <div class="separator" />

  <h2>Theoretical Background</h2>
  <p>Coming soon ... </p>
  <div class="separator" />
</Container>
<Footer {references} />
