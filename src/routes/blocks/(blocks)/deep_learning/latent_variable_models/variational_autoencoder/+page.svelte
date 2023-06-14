<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PythonCode from "$lib/PythonCode.svelte";
  import sampled from "./sampled.png";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Circle from "$lib/diagram/Circle.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Border from "$lib/diagram/Border.svelte";

  const references = [
    {
      author: "Diederik P. Kingma, Max Welling",
      title: "Auto-Encoding Variational Bayes",
      year: "2013",
    },
  ];

  const svgOffset = 40;
</script>

<svelte:head>
  <title>Variational Autoencoder - World4AI</title>
  <meta
    name="description"
    content="A variational autoencoder (VAE) is based on variational inference. Unlike a regular autoencoder, a variational autoencoder maps an input image to a latent variable distribution. We can use that distribution to sample new images."
  />
</svelte:head>

<Container>
  <h1>Variational Autoencoders</h1>
  <div class="separator" />
  <p>
    Variational autoencoders (VAE)<InternalLink id={1} type="reference" /> are latent
    variable models that have been relatively successfull in recent years. The mathematical
    background knowledge that is required to fully understand all intricacies of
    VAEs is quite extensive and can be intimidating for beginners. Therefore in this
    chapter we will mostly focus on the basics and the intuition, but we will also
    provide additional resources for you to study in case you would like to dive
    deeper into the theory. Fortunately the intuitive introduction is sufficient
    to implement a VAE from scratch in PyTorch, so you might skip the theory for
    now and return to it at a later point.
  </p>
  <p>
    VAEs generate new data in a two step process. First we sample a latent
    variable <Latex>{String.raw`\mathbf{z}`}</Latex> from the multivariate Gaussian
    distribution
    <Latex>{String.raw`\mathcal{N}(\mathbf{0}, \mathbf{I})`}</Latex>. Each value
    of the latent vector is distributed according to the standard normal
    distribution <Latex>{String.raw`\mathcal{N}(0, 1 )`}</Latex>
    and there is no interaction between the values within the vector. In the second
    step we generate the new sampe <Latex>{String.raw`\mathbf{X}`}</Latex> from the
    latent vector. The latent variable determines the characteristics of the data.
    In our implementation below for example we are going to generate new handwritten
    digits from Gaussian noise.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 200 300">
      <Arrow
        data={[
          { x: 30, y: 50 },
          { x: 30, y: 210 },
        ]}
        dashed={true}
        strokeDashArray="8 4"
        strokeWidth={2.5}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: 100, y: 50 },
          { x: 65, y: 50 },
        ]}
        dashed={true}
        strokeDashArray="8 4"
        strokeWidth={2.5}
        moving={true}
        speed={50}
      />
      <Circle
        x={30}
        y={50}
        r={25}
        text={"z"}
        class={"fill-red-100 stroke-2"}
        fontSize={30}
      />
      <Circle
        x={30}
        y={250}
        r={25}
        text={"X"}
        class={"fill-blue-100 stroke-2"}
        fontSize={30}
      />
      <Block
        x={145}
        y={50}
        width={90}
        height={55}
        text={String.raw`\mathcal{N}(0, 1)`}
        type="latex"
        fontSize="20"
        color="var(--main-color-3)"
      />
    </svg>
  </SvgContainer>
  <p>
    How can we make an autoencoder learn a model that can generate data from
    Gaussian noise? The autoencoder that we have discussed in the previous
    section follows a very simple approach: we map an image to a latent vector <Latex
      >{String.raw`\mathbf{z}`}</Latex
    > and reconstruct the original image <Latex>{String.raw`\mathbf{X}`}</Latex>
    from the latent variable. While this procedure is ideal for simple compression
    tasks, when it comes to generating new images the simple autoencoder is suboptimal.
    The latent space that is learned might not be smooth or continuous, so when you
    use a latent vector that deviates slightly from the samples that the decoder
    saw during training, you will end up with an invalid image. Moreover there is
    no built-in mechanism for sampling, yet the purpose of our task is to train a
    generative model that we can sample new images from.
  </p>
  <p>
    The variational autoencoder remedies those problems by mapping the image <Latex
      >{String.raw`\mathbf{X}`}</Latex
    > to a whole distribution of latent variables <Latex
      >{String.raw`q_{\phi}(\mathbf{z} | \mathbf{x})`}</Latex
    >. Our encoder neural network with parameters <Latex
      >{String.raw`\phi`}</Latex
    > produces two latent vectors: the vector with means
    <Latex>{String.raw`\mu_{\phi}`}</Latex> and the vector with variances <Latex
      >{String.raw`\sigma^2`}</Latex
    >. We use those values as input into an isotropic Gaussian distribution <Latex
      >{String.raw`\mathcal{N}(\mu_{\phi}, \sigma^2_{\phi} \mathbf{I})`}</Latex
    > (Gaussian with 0 covariance) and sample the latent variable <Latex
      >{String.raw`\mathbf{z}`}</Latex
    >.
  </p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 200 320">
      <Arrow
        data={[
          { x: 30, y: 40 },
          { x: 80, y: 160 },
        ]}
        dashed={true}
        strokeDashArray="8 4"
        strokeWidth={2}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: 170, y: 40 },
          { x: 120, y: 160 },
        ]}
        dashed={true}
        strokeDashArray="8 4"
        strokeWidth={2}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: 80, y: 180 },
          { x: 80, y: 205 },
        ]}
        dashed={true}
        strokeDashArray="8 4"
        strokeWidth={2}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: 120, y: 180 },
          { x: 120, y: 205 },
        ]}
        dashed={true}
        strokeDashArray="8 4"
        strokeWidth={2}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: 100, y: 220 },
          { x: 100, y: 275 },
        ]}
        dashed={true}
        strokeDashArray="8 4"
        strokeWidth={2}
        moving={true}
        speed={50}
      />
      <Block
        x={100}
        y={20}
        width={150}
        height={30}
        class="fill-yellow-100"
        fontSize={20}
        text="X"
      />
      <Block
        x={100}
        y={90}
        width={100}
        height={30}
        class="fill-blue-100"
        fontSize={20}
      />
      <Block
        x={80}
        y={160}
        width={30}
        height={30}
        class="fill-red-300"
        fontSize={20}
        text={String.raw`\mu`}
        type="latex"
      />
      <Block
        x={120}
        y={160}
        width={30}
        height={30}
        class="fill-red-300"
        fontSize={20}
        text={String.raw`\sigma`}
        type="latex"
      />
      <Block
        x={100}
        y={230}
        width={70}
        height={30}
        class="fill-red-300"
        fontSize={15}
        text={String.raw`\mathcal{N}(\mu, \sigma)`}
        type="latex"
      />
      <Block
        x={100}
        y={300}
        width={30}
        height={30}
        class="fill-red-300"
        fontSize={20}
        text={String.raw`z`}
      />
    </svg>
  </SvgContainer>
  <p>
    If you look at the graph above you might recognize the problem in the
    approach we described so far. How can we backpropagate our loss through the
    normal distribution? We can reframe our problem using the so called <Highlight
      >reparameterization trick</Highlight
    >. We rewrite the latent variable <Latex>{String.raw`\mathbf{z}`}</Latex> as
    the function of the mean, the standard deviation and the Gaussian noise <Latex
      >{String.raw`\epsilon \sim \mathcal{N}(0, 1)`}</Latex
    >.
  </p>
  <Latex
    >{String.raw`\mathbf{z} = \boldsymbol \mu + \boldsymbol \sigma \odot \boldsymbol \epsilon \text{, where } \odot \text{ is elementwise multiplication}`}</Latex
  >.
  <p>
    This rewriting does not change the fact that the latent vector <Latex
      >{String.raw`\mathbf{z}`}</Latex
    > is distributed according to a multivariate Gaussian with mean <Latex
      >{String.raw`\boldsymbol \mu`}</Latex
    >, but we can backpropagate through the mean and the variance and treat <Latex
      >{String.raw`\boldsymbol \epsilon`}</Latex
    > as some constant that does not need to be optimized.
  </p>
  <p>
    Once we start to implement the encoder in PyTorch we will notice a problem
    with our approach. The ecoder neural network produces <Latex
      >{String.raw`\boldsymbol \mu`}</Latex
    > and <Latex>{String.raw`\boldsymbol \sigma`}</Latex> as the output of a linear
    layer. A linear layer can theoretically produce positive and negative numbers.
    While this is not a problem for the mean, the variance and the standard deviation
    are always positive. To circumvent this problem we will assume that the linear
    layer generates <Latex>{String.raw`\log \boldsymbol \sigma^2`}</Latex>,
    which can be positive or negative. To transform the logarithm of the
    variance back to standard deviation we can use the following equality: <Latex
      >{String.raw`\large \sigma = e^{0.5\times \log\sigma^2}`}</Latex
    >.
  </p>

  <p>
    The decoder of the variation autoencoder works in the exact same way as the
    one introduced in the previous section: given a latent vector <Latex
      >{String.raw`\mathbf{z}`}</Latex
    > reconstruct the original image as close as possible.
  </p>
  <p>The complete variational autoencoder looks as follows.</p>
  <SvgContainer maxWidth={"700px"}>
    <svg viewBox="0 0 470 250">
      <!-- Encoder Arrows -->
      <Arrow
        data={[
          { x: svgOffset + 12.5 + 5, y: 90 - 75 },
          { x: svgOffset + 80 * 2 - 12.5 - 10, y: 90 - 20 },
        ]}
        dashed={true}
        strokeDashArray="4 4"
        strokeWidth={1.5}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: svgOffset + 12.5 + 5, y: 90 + 75 },
          { x: svgOffset + 80 * 2 - 12.5 - 10, y: 90 + 20 },
        ]}
        dashed={true}
        strokeDashArray="4 4"
        strokeWidth={1.5}
        moving={true}
        speed={50}
      />

      <!-- sampling arrows -->
      <Arrow
        data={[
          { x: svgOffset + 80 * 2 + 12.5, y: 60 },
          { x: svgOffset + 80 * 2 + 60, y: 80 },
        ]}
        dashed={true}
        strokeDashArray="4 4"
        strokeWidth={1.5}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: svgOffset + 80 * 2 + 12.5, y: 120 },
          { x: svgOffset + 80 * 2 + 60, y: 100 },
        ]}
        dashed={true}
        strokeDashArray="4 4"
        strokeWidth={1.5}
        moving={true}
        speed={50}
      />
      <Arrow
        data={[
          { x: svgOffset + 80 * 2 + 12.5, y: 225 },
          { x: svgOffset + 80 * 2 + 80, y: 225 },
          { x: svgOffset + 80 * 2 + 80, y: 115 },
        ]}
        dashed={true}
        strokeDashArray="4 4"
        strokeWidth={1.5}
        moving={true}
        speed={50}
      />

      <!-- Encoder -->
      <Block
        x={svgOffset + 0}
        y={90}
        width={25}
        height={150}
        text="X"
        type="latex"
        fontSize="20"
        class="fill-yellow-100"
      />
      <Block
        x={svgOffset + 80}
        y={90}
        width={25}
        height={100}
        class="fill-blue-100"
      />

      <!-- mu, sigma and epsilon -->
      <Block
        x={svgOffset + 80 * 2}
        y={60}
        width={25}
        height={30}
        text="\boldsymbol \mu"
        type="latex"
        fontSize="20"
        class="fill-red-300"
      />
      <Block
        x={svgOffset + 80 * 2}
        y={120}
        width={25}
        height={30}
        text="\boldsymbol \sigma"
        type="latex"
        fontSize="20"
        class="fill-red-300"
      />
      <Block
        x={svgOffset + 80 * 2}
        y={225}
        width={25}
        height={30}
        text="\boldsymbol \epsilon"
        type="latex"
        fontSize="20"
        class="fill-red-300"
      />

      <g transform="translate(80, 0)">
        <!-- Decoder Arrows -->
        <Arrow
          data={[
            { x: svgOffset + 80 * 2 + 12.5 + 5, y: 90 - 20 },
            { x: svgOffset + 80 * 4 - 12.5 - 10, y: 90 - 75 },
          ]}
          dashed={true}
          strokeDashArray="4 4"
          strokeWidth={1.5}
          moving={true}
          speed={50}
        />
        <Arrow
          data={[
            { x: svgOffset + 80 * 2 + 12.5 + 5, y: 90 + 20 },
            { x: svgOffset + 80 * 4 - 12.5 - 10, y: 90 + 75 },
          ]}
          dashed={true}
          strokeDashArray="4 4"
          strokeWidth={1.5}
          moving={true}
          speed={50}
        />

        <!-- Bottleneck -->
        <Block
          x={svgOffset + 80 * 2}
          y={90}
          width={25}
          height={30}
          text={String.raw`\mathbf{z}`}
          type="latex"
          fontSize="20"
          class="fill-red-300"
        />

        <!-- Decoder -->
        <Block
          x={svgOffset + 80 * 3}
          y={90}
          width={25}
          height={100}
          class="fill-blue-200"
        />
        <Block
          x={svgOffset + 80 * 4}
          y={90}
          width={25}
          height={150}
          text={String.raw`\hat{X}`}
          type="latex"
          fontSize="20"
          class="fill-yellow-100"
        />
      </g>
      <!-- Borders -->
      <Border x={10} y={5} width={140} height={180} />
      <Border x={300} y={5} width={160} height={180} />
      <svg /></svg
    ></SvgContainer
  >
  <p>
    The last remaining puzzle is the loss function that we use to train a VAE,
    which consists of two parts: the <Highlight>reconstruction loss</Highlight> and
    a <Highlight>regularizer term</Highlight>.
  </p>
  <p>
    The reconstruction loss we are going to use is the mean squared error
    between each pixel of the original image <Latex>{String.raw`X`}</Latex> and the
    reconstructed image <Latex>{String.raw`X'`}</Latex>. This is the exact same
    loss that we used with the regular autoencoder. Be aware that sometimes the
    cross-entropy loss is used to measure the reconstruction quality, but for
    MNIST MSE works great.
  </p>
  <p>
    The regularizer (Kullback-Leibler divergence) on the other hand tries to
    push the mean and the variance that the encoder outputs close to 0 and 1
    respectively. This can be achieved by minimizing the following expression: <Latex
      >{String.raw`- \dfrac{1}{2} \sum^n_i (1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2)`}</Latex
    >, where n is the size of the latent variable vector. Try to replace the
    mean by 0 and the variance by 1 in the above expression and see what
    happens. The loss goes to 0. This regularizer allows us to sample from the
    isotropic Gaussian with mean vector of 0 and the standard deviation vector
    of 1 and generate realistic images from this Gaussian noise.
  </p>
  <p>
    Below is the implementation of the VAE. There should not be any unexptected
    code snippets at this point.
  </p>
  <PythonCode
    code={`class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.mu = nn.Linear(in_features=1600, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=1600, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=1600),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 5, 5)),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3),
            nn.Sigmoid(),
        )

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = self.encoder(x)
        mu, sigma = self.mu(x), torch.exp(self.log_var(x) / 2)
        epsilon = torch.randn_like(mu, device=DEVICE)
        x = mu + sigma * epsilon
        x = self.decoder(x)
        return x, mu, sigma`}
  />
  <p>
    The <code>train</code> function on the other hand still holds a little
    surprise for us. When we calculate our full loss we scale the
    <code>reconstruction_loss</code> by a factor of 0.01. The MSE loss and the regularizer
    (kl_loss) are on different scales. In our implementation the MSE loss can be
    100 times larger than the regularizer. If we do not scale MSE by 0.01 the improvement
    of the regularizer will progress slowly and the sampling quality from Gaussian
    noise will deteriorate.
  </p>
  <PythonCode
    code={`def train(num_epochs, train_dataloader, model, criterion, optimizer):
    history = {"reconstruction_loss": [], "kl_loss": [], "full_loss": []}
    model.to(DEVICE)
    for epoch in range(num_epochs):
        num_batches = 0
        history["reconstruction_loss"] = []
        history["kl_loss"] = []
        history["full_loss"] = []

        for batch_idx, (features, _) in enumerate(train_dataloader):
            model.train()
            num_batches += 1

            features = features.to(DEVICE)

            # Forward Pass
            output, mu, sigma = model(features)

            # Calculate Loss

            # RECONSTRUCTION LOSS
            reconstruction_loss = criterion(output, features)
            reconstruction_loss = reconstruction_loss.mean()

            history["reconstruction_loss"].append(reconstruction_loss.cpu().item())

            # KL LOSS
            kl_loss = -0.5 * (1 + (sigma**2).log() - mu**2 - sigma**2).sum(dim=1)
            kl_loss = kl_loss.mean()

            history["kl_loss"].append(kl_loss.cpu().item())

            # FULL LOSS
            loss = 0.01 * reconstruction_loss + kl_loss

            history["full_loss"].append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        reconstruction_loss, kl_loss, full_loss = (
            sum(history["reconstruction_loss"]) / num_batches,
            sum(history["kl_loss"]) / num_batches,
            sum(history["full_loss"]) / num_batches,
        )

        print(
            f"Epoch: {epoch+1:>2}/{num_epochs} | Reconstruction Loss: {reconstruction_loss:.5f} | KL Loss: {kl_loss:.5f} | Full Loss: {full_loss:.5f}"
        )
`}
  />
  <p>
    To sample images, we first sample random noise from the standad normal
    distribution and pass the noise throught the decoder. After training the
    model for 50 epochst we get the following results. The quality is not ideal,
    but for the most part we can recognize the digits.
  </p>
  <PythonCode
    code={`num_images = 6
    with torch.inference_mode():
        z = torch.randn(num_images, LATENT_DIM).to(DEVICE)
        images = vae.decode(z)
        fig = plt.figure(figsize=(15, 4))
        for i, img in enumerate(images):
            fig.add_subplot(1, 6, i + 1)
            img = img.squeeze().cpu().numpy()
            plt.imshow(img, cmap="gray")
            plt.axis("off")
    plt.savefig("sampled.png", bbox_inches="tight")
`}
  />
  <img src={sampled} alt="Handwritten digits sampled from a VAE" />
  <p>
    If you would like to dive deeper into the mathematical derivations of the
    VAE, there are a couple of sources we would recommend. This <a
      href="https://lilianweng.github.io/posts/2018-08-12-vae/"
      target="_blank">blog post</a
    >
    by Lilian Weng contains a good overview of different autoencoders. Additionally
    we would recommend this
    <a href="https://www.youtube.com/watch?v=7Pcvdo4EJeo" target="_blank"
      >YouTube video</a
    > by DeepMind, which provides a good introduction into the theory of latent variable
    models.
  </p>
  <div class="separator" />
</Container>
<Footer {references} />
