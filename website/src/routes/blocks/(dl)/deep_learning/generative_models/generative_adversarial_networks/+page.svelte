<script>
  import Container from "$lib/Container.svelte"; 
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";
  
  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Border from "$lib/diagram/Border.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const references = [
    {
        author: "Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua",
        title: "Generative Adversarial Nets",
        journal: "Advances in Neural Information Processing Systems",
        year: "2014",
        volume: "27",
    }
  ]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Generative Adversarial Networks</title>
  <meta
    name="description"
    content="Generative adversarial networks are a pair of neural networks, that compete againts each other. The generator creates fake images, while the discriminator tries to separate real from fake images."
  />
</svelte:head>

<h1>Generative Adversarial Networks</h1>
<div class="separator"></div>

<Container>
  <h2>Vanilla Generative Adversarial Networks</h2>
  <p>Generative adversarial networks<InternalLink id={1} type="reference"/> (GAN) were designed to train a generative model through an adversarial process. The authors of the GAN paper provided the following explanation to exemplify how this process looks like.</p> 
  <p class="info">"There is a team of counterfeiters, trying to produce fake currency and use it without detection, while the the police is trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles."</p>
  <p>The counterfeiters and the police can be thought of as two separate fully connected neural networks, that are optimized using two opposing goals and can thought of as adversaries in a two player game. In our case the counterfeiters generate new data (like images) therefore this network is called the generator model <Latex>{String.raw`G`}</Latex>. The other (policing) network is designed to distinguished between real and fake/generated data and is therefore called the discriminator <Latex>{String.raw`D`}</Latex>.</p>
  <SvgContainer maxWidth="800px">
    <svg viewBox="0 0 440 140">
      <Block x=20 y=110 width=20 height=20 text="z" fontSize=15 type="latex"/>   
      <Block x=100 y=110 width=40 height=40 text="G" fontSize=15 type="latex" color="var(--main-color-3)" />   
      <Block x=200 y=30 width=40 height=40 text="x" fontSize=15 type="latex" />   
      <Block x=200 y=110 width=40 height=40 text="G(z)" fontSize=15 type="latex" />   
      <Block x=300 y=70 width=40 height=40 text="D" fontSize=15 type="latex" color="var(--main-color-3)"/>   
      <Block x=400 y=70 width=50 height=20 text="p(true)" fontSize=10 type="latex" />   
      <Border x=5 y=80 width=220 height=55 />
      <Border x=170 y=2 width=260 height=135 />

      <!-- titles -->
      <Block x=100 y=60 width=70 height=20 text="Generator" fontSize=10 color="var(--main-color-1)"/>
      <Block x=380 y=20 width=80 height=20 text="Discriminator" fontSize=10 color="var(--main-color-1)"/>

      <!-- moving arrows -->
      <Arrow data={[{x:35, y:110}, {x:70, y:110}]} strokeWidth=1.5 dashed={true} strokeDashArray="5, 5" moving={true} speed=50/>
      <Arrow data={[{x:120, y:110}, {x:175, y:110}]} strokeWidth=1.5 dashed={true} strokeDashArray="5, 5" moving={true} speed=50/>
      <Arrow data={[{x:220, y:110}, {x:300, y:110}, {x:300, y: 97}]} strokeWidth=1.5 dashed={true} strokeDashArray="5, 5" moving={true} speed=50/>
      <Arrow data={[{x:220, y:30}, {x:300, y:30}, {x:300, y: 44}]} strokeWidth=1.5 dashed={true} strokeDashArray="5, 5" moving={true} speed=50/>
      <Arrow data={[{x:320, y:70}, {x:370, y:70} ]} strokeWidth=1.5 dashed={true} strokeDashArray="5, 5" moving={true} speed=50/>
    </svg>
  </SvgContainer>

  <p>We start the generation process by drawing Gaussian noise from the standard normal distribution<Latex>{String.raw`z \sim N(0,1)`}</Latex>. In the next step we generate a fake image by feeding a noise/latent vector <Latex>{String.raw`\mathbf{z}`}</Latex> into the generator neural network <Latex>{String.raw`G`}</Latex>. The discriminator neural network <Latex>{String.raw`D`}</Latex> on the other hand receveis a batch of real data <Latex>{String.raw`\mathbf{x}`}</Latex> and a batch of fake data <Latex>{String.raw`G(z)`}</Latex> and needs to predict the probability of the data to be real.</p>
  <p>The intuition why this process works goes as follows. At the beginnig the discriminator can not differentiate between true and fake images, but the discriminator faces a relatively straighforward classification task, which is not that hard to learn. Once classification accuracy increases the generator needs to learn to generate better images in order to fool the discriminator, which in turn forces the discriminator to get better and so the arms race keeps continues until the generator creates images that look real and the discriminator guesses with a probability of 50% whether the images are real or not. Unfortunately the reality is more complicated. GANs are notoriously hard to train, as we will discover in our practice section. In fact there are hundreds of GANs architectures which were designed to improve on the original GAN architecture. Some of them we will discuss in a separate section.</p>
  <p>The generator and the discriminator play a so called min-max game with the following value function <Latex>{String.raw`V`}</Latex>, which the generator tries to minimize and the discriminator tries to maximize.</p>
  <Latex>{String.raw`\large \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(\mathbf{x})] + \mathbb{E}_{ z \sim p_{z}(z)}[\log(1 - D(G(\mathbf{z})))]`}</Latex>  
  <p>Now let's see why this is the case starting with the discriminator. If the discriminator faces real data <Latex>{String.raw`\mathbf{x}`}</Latex>, it makes sense to maximize <Latex>{String.raw`D(x)`}</Latex> indicates the probability of the data to be real. If the discriminator faces fake data <Latex>{String.raw`G(\mathbf{z})`} </Latex>, the discriminator will try to reduce the probability <Latex>{String.raw`D(G(\mathbf{z}))`} </Latex>, thereby increasing <Latex>{String.raw`\log(1 - D(G(\mathbf{z})))`} </Latex>. The generator does the exact opposite. Its goal is to fool the discriminator and to increase the probability of the fake data to be seen as real, which can be achieved by maximizing <Latex>{String.raw`D(G(\mathbf{z}))`}</Latex>.</p>
  <div class="separator"></div>
  <h2>DCGAN</h2>
  <div class="separator"></div>
</Container>
<Footer {references} />

