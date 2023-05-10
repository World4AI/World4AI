<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Border from "$lib/diagram/Border.svelte";

  import outputs from "./outputs.png";

  const svgOffset = 40;
</script>

<svelte:head>
  <title>Autoencoders - World4AI</title>
  <meta
    name="description"
    content="An autoencoder uses an encoder to map its input to latent space and uses a decoder to transform the latent variable back into the original image. The bottleneck is usually a lower dimensional vector that the input space, therefore an autoencoder compresses the input information."
  />
</svelte:head>

<Container>
  <h1>Autoencoders</h1>
  <div class="separator" />
  <p>
    A simple <Highlight>autoencoder</Highlight> like the one we will study in this
    section is not commonly used for generative models, but the knowledge of autoencoders
    is fundamental to studying more powerful architectures called variational autoencoders.
  </p>
  <Alert type="info">
    An autoencoder is a neural network architecture, that maps an input into a
    lower dimensional space and reconstructs the original input from the lower
    dimensional space.
  </Alert>
  <p>
    We use some variable <Latex>{String.raw`\mathbf{X}`}</Latex> as an input into
    an autoencoder. This could be a vector input in a fully connected neural network,
    but an autoencoder is also often used with images in combination with a convolutional
    neural network. An autoencoder is trained in an unsupervised way, without using
    any additional labels. Specifically the input and the output of an autoencoder
    are identical. So if we use an image as an input, we expect the output of the
    neural network, <Latex>{String.raw`\mathbf{X}'`}</Latex>, to be as close as
    possible to the original input image.
  </p>
  <p>
    An autoencoder consists of two components: an <Highlight>encoder</Highlight>
    and a <Highlight>decoder</Highlight>.
  </p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 400 230" class="border">
      <!-- Arrows -->
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
      <!-- Encoder -->
      <Block
        x={svgOffset + 0}
        y={90}
        width={25}
        height={150}
        text={String.raw`\mathbf{X}`}
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
      <!-- Bottleneck -->
      <Block
        x={svgOffset + 80 * 2}
        y={90}
        width={25}
        height={40}
        text={String.raw`\mathbf{z}`}
        type="latex"
        fontSize="20"
        class="fill-red-400"
      />
      <!-- Decoder -->
      <Block
        x={svgOffset + 80 * 3}
        y={90}
        width={25}
        height={100}
        class="fill-blue-300"
      />
      <Block
        x={svgOffset + 80 * 4}
        y={90}
        width={25}
        height={150}
        text={String.raw`\mathbf{\hat{X}}`}
        type="latex"
        fontSize="20"
        class="fill-lime-100"
      />
      <!-- Borders -->
      <Border x={10} y={5} width={140} height={180} />
      <Border x={250} y={5} width={140} height={180} />
      <!-- Labels -->
      <Block
        x={50}
        y={210}
        width={80}
        height={25}
        text="Encoder"
        class="fill-slate-300"
        fontSize="15"
      />
      <Block
        x={200}
        y={40}
        width={80}
        height={20}
        text="Bottleneck"
        fontSize="12"
        class="fill-green-100"
      />
      <Block
        x={350}
        y={210}
        width={80}
        height={25}
        text="Decoder"
        class="fill-slate-300"
        fontSize="15"
      />
      <svg /></svg
    ></SvgContainer
  >
  <p>
    The encoder takes the input image <Latex>{String.raw`\mathbf{X}`}</Latex> and
    produces the latent variable vector <Latex>{String.raw`\mathbf{z}`}</Latex>.
    With each layer we keep decreasing the dimensionality of hidden values until
    we arrive at the so called <Highlight>bottleneck</Highlight>: the last layer
    of the encoder that outputs a relatively low dimensional vector. By
    decreasing the dimensionality we compress the information that is contained
    in the input, until we reach the highest compression point with at the
    bottleneck.
  </p>
  <p>
    The decoder does the exact opposite. It takes the latent variable vector <Latex
      >{String.raw`\mathbf{z}`}</Latex
    > as an input and tries to uncompress the information into the original space.
    With each layer the dimnesionality of the vector increases, until the neural
    network reaches the original dimensionality.
  </p>
  <p>
    The neural network has to learn to produce <Latex
      >{String.raw`\mathbf{\hat{X}}`}</Latex
    >, that is as close as possible to the input. As we squeeze a high
    dimensional image into a low dimensional vector, the compression is lossy
    and the output image might lose some detail, but if the network is
    expressive enough, the loss won't be dramatic.
  </p>
  <p>
    Intuitively we could argue, that the neural network removes all the
    unnecessary noise, until only the relevant characteristics of the data that
    are contained in the latent space are left.
  </p>
  <p>
    In our implementation below we create two <code>nn.Sequential</code> modules:
    one for the encoder and the other for the decoder. The encoder uses a stack of
    convolutional layers and a single fully connected layer to compress MNIST images
    into a 10 dimensional vector. The decoder uses transposed convolutions to decompress
    the image into a 1x28x28 dimensional tensor. The last layer of the decoder is
    a sigmoid activation function, because we scale the original images between 0
    and 1.
  </p>
  <PythonCode
    code={`class Autoencoder(nn.Module):
    def __init__(self):
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
            nn.Linear(in_features=1600, out_features=10),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=10, out_features=1600),
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x`}
  />
  <p>
    We use the mean squared error to minimize the distance between the original
    and reconstructed images and end up with the following results after 20
    epochs of training.
  </p>
  <img src={outputs} alt="Original and generated images" />
  <p>
    The top row depicts 10 original images from the validation dataset and the
    bottom row shows the recconstructed images that went through a 10
    dimensional bottleneck. While the reconstruction made the images blurry, the
    quality is outstanding, especially given that we compressed those images
    from a 28x28 dimensional vector into a 10 dimensional vector.
  </p>
  <p>
    While compressing the images into a lower dimensional space is an important
    task, we would like to use an autoencoder to generate new images and we can
    theoretically utilize the decoder for that purpose. Think about it, the
    decoder takes the latent variable vector and produces an image from a low
    dimensional space. The problem with this vanilla autoencoder is that we have
    no built-in mechanism to sample new latent variables from which we could
    generate the images. A variational autoencoder on the other hand is
    specifically designed for the purpose to generate new samples. We will study
    and utilze this architecture in the next section.
  </p>
  <div class="separator" />
</Container>
