<script>
  import { onMount } from "svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";
  import Alert from "$lib/Alert.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Border from "$lib/diagram/Border.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  import results from "./results.png";

  const references = [
    {
      author:
        "Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua",
      title: "Generative Adversarial Nets",
      journal: "Advances in Neural Information Processing Systems",
      year: "2014",
      volume: "27",
    },
  ];

  //show either true image or generated image
  let inp = 0;
  onMount(() => {
    const interval = setInterval(() => {
      if (inp === 0) {
        inp = 1;
      } else if (inp === 1) {
        inp = 0;
      }
    }, 2000);

    return () => clearInterval(interval);
  });
</script>

<svelte:head>
  <title>Generative Adversarial Networks Introduction - World4AI</title>
  <meta
    name="description"
    content="Generative adversarial networks are a pair of neural networks, that compete againts each other. The generator creates fake images, while the discriminator tries to separate real from fake images."
  />
</svelte:head>

<Container>
  <h1>Generative Adversarial Networks</h1>
  <div class="separator" />
  <p>
    Generative adversarial networks (commonly known as GANs) is a family of
    generative models, that were designed to be trained through an adversarial
    process. The easiest way to explain what that actually means is by looking
    at a quote from the paper by Goodfellow et. al. that originally introduced
    the GAN architecture <InternalLink id={1} type="reference" />.
  </p>
  <Alert type="quote">
    There is a team of counterfeiters, trying to produce fake currency and use
    it without detection, while the the police is trying to detect the
    counterfeit currency. Competition in this game drives both teams to improve
    their methods until the counterfeits are indistiguishable from the genuine
    articles.
  </Alert>
  <p>
    The counterfeiters and the police are actually two separate fully connected
    neural networks. The counterfeiters neural network is called the <Highlight
      >generator</Highlight
    >
    <Latex>{String.raw`G`}</Latex>. This model takes a vector of random noise <Latex
      >{String.raw`\mathbf{z}`}</Latex
    > and produces an image <Latex>{String.raw`G(\mathbf{z})`}</Latex>.
  </p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 300 400">
      {#each Array(4) as _, idx}
        <Block
          x={150}
          y={15 + idx * 18}
          width={15}
          height={15}
          class="fill-blue-300"
        />
      {/each}
      <Arrow
        data={[
          { x: 150, y: 80 },
          { x: 150, y: 160 },
        ]}
        strokeWidth={2.5}
        dashed={true}
        strokeDashArray="8 8"
        moving={true}
      />
      <Block
        x={150}
        y={200}
        width={50}
        height={50}
        text={String.raw`G`}
        fontSize={25}
        class="fill-yellow-200"
        type="latex"
      />

      {#each Array(4) as _, xIdx}
        {#each Array(4) as _, yIdx}
          <Block
            x={125 + xIdx * 18}
            y={330 + yIdx * 18}
            width={15}
            height={15}
            class="fill-lime-200"
          />
        {/each}
      {/each}
      <g transform="translate(0 150)">
        <Arrow
          data={[
            { x: 150, y: 80 },
            { x: 150, y: 160 },
          ]}
          strokeWidth={2.5}
          dashed={true}
          strokeDashArray="8 8"
          moving={true}
        />
      </g>
    </svg>
  </SvgContainer>
  <p>
    To generate an image we can simply sample a vector <Latex
      >{String.raw`\mathbf{z}`}</Latex
    > from the standard normal distribution and pass it through the function <Latex
      >G</Latex
    >. The original GAN implementation is based on fully connected neural
    networks, therefore we will have to reshape the image into a 2d tensor at a
    later step.
  </p>
  <PythonCode
    code={`class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(HIDDEN_SIZE, IMG_SIZE),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)
`}
  />
  <p>
    Our implementation is fairly simple. We use two fully connected layers with
    a leaky <code>ReLU</code> inbetween and a tanh as the output. It is fairly
    common to use
    <code>tanh</code> for the output of the generator, therefore we will need to
    scale the traininig data between -1 and 1.
  </p>
  <p>
    The policing network, the <Highlight>discriminator</Highlight>
    <Latex>{String.raw`D`}</Latex>, is designed to distinguished between real
    and generated data.
  </p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 300 400">
      {#if inp === 0}
        <Block
          x={50}
          y={35}
          width={60}
          height={60}
          class="fill-blue-300"
          text={String.raw`\mathbf{x}`}
          type="latex"
          fontSize={20}
        />
        <Arrow
          data={[
            { x: 50, y: 80 },
            { x: 140, y: 160 },
          ]}
          strokeWidth={2.5}
          dashed={true}
          strokeDashArray="8 8"
          moving={true}
        />
      {/if}
      {#if inp === 1}
        <Block
          x={250}
          y={35}
          width={60}
          height={60}
          class="fill-blue-300"
          type="latex"
          text={String.raw`G(\mathbf{z})`}
          fontSize={20}
        />
        <Arrow
          data={[
            { x: 250, y: 80 },
            { x: 160, y: 160 },
          ]}
          strokeWidth={2.5}
          dashed={true}
          strokeDashArray="8 8"
          moving={true}
        />
      {/if}
      <Block
        x={150}
        y={200}
        width={50}
        height={50}
        text={String.raw`D`}
        fontSize={25}
        class="fill-yellow-200"
        type="latex"
      />
      <g transform="translate(0 150)">
        <Arrow
          data={[
            { x: 150, y: 80 },
            { x: 150, y: 160 },
          ]}
          strokeWidth={2.5}
          dashed={true}
          strokeDashArray="8 8"
          moving={true}
        />
      </g>
      <Block
        x={150}
        y={360}
        width={50}
        height={50}
        text={inp === 0 ? "T" : "F"}
        fontSize={25}
        class={inp === 0 ? "fill-green-300" : "fill-red-400"}
      />
    </svg>
  </SvgContainer>
  <p>
    If the input is a true image <Latex>{String.raw`\mathbf{x}`}</Latex> the discriminator
    is expected to generate a value close to 1, otherwise it should generate a value
    close to 0. Basically the discriminator generates a probability that indicates
    if the input is a real image or not.
  </p>
  <PythonCode
    code={`class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(IMG_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.discriminator(x)`}
  />
  <p>
    Normally we would need to use a sigmoid as the output layer of the
    discriminator, but <code> nn.BCEWithLogitsLoss()</code> already takes care of
    that, as it combines sigmoid with binary cross-entropy.
  </p>
  <p>
    During the training process we combine the generator and the discriminator
    and train both jointly. We start the generation process by drawing Gaussian
    noise from the standard normal distribution<Latex
      >{String.raw`z \sim N(0,1)`}</Latex
    >. Next we generate fake images by feeding a noise vector (also called
    latent vector) <Latex>{String.raw`\mathbf{z}`}</Latex> into the generator neural
    network
    <Latex>{String.raw`G`}</Latex>. The discriminator neural network <Latex
      >{String.raw`D`}</Latex
    > receveis a batch of real data <Latex>{String.raw`\mathbf{x}`}</Latex> and a
    batch of fake data <Latex>{String.raw`G(\mathbf{z})`}</Latex> and needs to predict
    the probability that the data is real.
  </p>
  <SvgContainer maxWidth="230px">
    <svg viewBox="0 0 230 490">
      <Block
        x="40"
        y="20"
        width="20"
        height="20"
        text="z"
        fontSize="15"
        type="latex"
      />
      <Block
        x="40"
        y="120"
        width="40"
        height="40"
        text="G"
        fontSize="15"
        type="latex"
        class="fill-yellow-200"
      />
      <Block
        x="40"
        y="220"
        width="40"
        height="40"
        text="G(z)"
        fontSize="15"
        type="latex"
      />
      <Block
        x="200"
        y="220"
        width="40"
        height="40"
        text="x"
        fontSize="15"
        type="latex"
      />
      <Block
        x="120"
        y="320"
        width="40"
        height="40"
        text="D"
        fontSize="15"
        type="latex"
        class="fill-yellow-200"
      />
      <Block
        x="120"
        y="450"
        width="30"
        height="30"
        text="p"
        fontSize="15"
        type="latex"
      />
      <Border x="10" y="5" width="60" height="250" />
      <Border x="5" y="185" width="220" height="300" />

      <!-- moving arrows -->
      <Arrow
        data={[
          { x: 40, y: 30 },
          { x: 40, y: 90 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="5, 5"
        moving={true}
        speed="50"
      />
      <Arrow
        data={[
          { x: 40, y: 140 },
          { x: 40, y: 190 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="5, 5"
        moving={true}
        speed="50"
      />
      <Arrow
        data={[
          { x: 40, y: 240 },
          { x: 40, y: 320 },
          { x: 90, y: 320 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="5, 5"
        moving={true}
        speed="50"
      />
      <Arrow
        data={[
          { x: 200, y: 240 },
          { x: 200, y: 320 },
          { x: 145, y: 320 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="5, 5"
        moving={true}
        speed="50"
      />
      <Arrow
        data={[
          { x: 120, y: 340 },
          { x: 120, y: 425 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="5, 5"
        moving={true}
        speed="50"
      />
    </svg>
  </SvgContainer>
  <p>
    The intuition why this process works goes as follows. At the beginnig the
    discriminator can not differentiate between true and fake images, but it
    faces a relatively straighforward classification task. Once classification
    accuracy increases the generator needs to learn to generate better images in
    order to fool the discriminator, which in turn forces the discriminator to
    get better and so the arms race continues until the generator creates images
    that look real and the discriminator generates a probability of 50%, because
    the generated images look realistic can not be distinguished from real
    images. Unfortunately the reality is more complicated. GANs are notoriously
    hard to train. In fact there are dozens of GANs architectures which were
    designed to improve on the original GAN architecture. We will encounter
    several of those architectures as move through this chapter.
  </p>
  <p>
    In reality during training the generator and the discriminator play a so
    called min-max game with the following value function <Latex
      >{String.raw`V`}</Latex
    >.
  </p>
  <Latex
    >{String.raw`\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(\mathbf{x})] + \mathbb{E}_{ z \sim p_{z}(z)}[\log(1 - D(G(\mathbf{z})))]`}</Latex
  >
  <p>
    The generator tries to minimize this function, while the discriminator tries
    to maximize this function. But why does this make sense? If the
    discriminator faces real data <Latex>{String.raw`\mathbf{x}`}</Latex>, it
    makes sense to maximize <Latex>{String.raw`D(\mathbf{x})`}</Latex>, because <Latex
      >{String.raw`D(\mathbf{x})`}</Latex
    > indicates the probability of the data to be real. If the discriminator faces
    fake data <Latex>{String.raw`G(\mathbf{z})`}</Latex>, the discriminator will
    try to reduce the probability <Latex>{String.raw`D(G(\mathbf{z}))`}</Latex>,
    thereby increasing <Latex>{String.raw`\log(1 - D(G(\mathbf{z})))`}</Latex>.
    The generator does the exact opposite. Its goal is to fool the discriminator
    and to increase the probability of the fake data to be seen as real, which
    can be achieved by maximizing <Latex>{String.raw`D(G(\mathbf{z}))`}</Latex>.
  </p>
  <p>
    Deep learning frameworks work with gradient descent, yet we are expected to
    maximize the value function from the point of the discriminator, so let's
    transform the above problem into a format that will be compatible with
    PyTorch. This is relatively straightforward, because maximizing an
    expression and miniziming a negative expression should lead to the same
    results.
  </p>
  <Latex
    >{String.raw`L_n = - [ y_n \cdot \log D(x_n) + (1 - y_n) \cdot \log (1 - D(G(z_n))]`}</Latex
  >
  <p>
    We essentially frame the problem as a binary cross-entropy loss. If the
    discriminator faces a true image, the loss will collapse to <Latex
      >-\log D(x_n)</Latex
    >. If the discriminator faces a fake image, the loss will collapse to <Latex
      >-[\log (1 - D(G(z_n))]</Latex
    >.
  </p>
  <p>
    The generator is already framed as a minimization problem, yet we still face
    a practical problem. Especially at the beginning of the training the value <Latex
      >\log(1 - D(G(z))</Latex
    > will be close to 1 as the discriminator will have it easy to distinguish between
    real and fake images. <Latex>D(G(z))</Latex> and its gradient will be close to
    0 and the generator will have a hard time training. The authors therefore suggest
    to turn the problem into a maximization problem: maximize <Latex
      >\log D(G(z))</Latex
    >. This leads to the same result, but as mentioned before PyTorch needs a
    problem to be framed in terms of gradient descent, so we minimize <Latex
      >-\log D(G(z))</Latex
    >. While we are optimizing towards the same weights and biases, the
    gradients are much larger at the beginning when the discriminator has an
    easy time. Try to think through a couple of examples to make sure you
    understand why the new expression is better for the training process.
  </p>
  <PythonCode
    code={`for epoch in range(NUM_EPOCHS):
    dis_loss_col = []
    gen_loss_col = []
    for batch_idx, (features, _) in enumerate(dataloader):
        real_images = features.view(-1, IMG_SIZE).to(DEVICE)

        # generate fake images from standar normal distributed latent vector
        latent_vector = torch.randn(BATCH_SIZE, LATENT_SIZE, device=DEVICE)
        fake_imgs = generator(latent_vector)

        # calculate logits for true and fake images
        fake_logits = discriminator(fake_imgs.detach())
        real_logits = discriminator(real_images)

        # calculate discriminator loss
        dis_real_loss = criterion(real_logits, torch.ones(BATCH_SIZE, 1, device=DEVICE))
        dis_fake_loss = criterion(
            fake_logits, torch.zeros(BATCH_SIZE, 1, device=DEVICE)
        )
        dis_loss = dis_real_loss + dis_fake_loss

        # optimize the discriminator
        dis_optim.zero_grad()
        dis_loss.backward()
        dis_optim.step()

        # calculate generator loss
        gen_loss = criterion(
            discriminator(fake_imgs), torch.ones(BATCH_SIZE, 1, device=DEVICE)
        )

        # optimize the generator
        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()`}
  />
  <p>
    It might not be obvious at first glance, but when we calculate the generator
    loss we flip the labels: we use 1's instead of 0's. That trick transforms
    the generator loss into the right format. Try to understand why that works.
  </p>
  <p>
    Below are the results that were generated by our simple GAN after 100
    epochs. There is definetely room for improvement and we will look at better
    GAN architectures in the following sections of this chapter.
  </p>
  <div class="flex justify-center">
    <img src={results} alt="MNIST generated by a GAN" />
  </div>
</Container>
<Footer {references} />
