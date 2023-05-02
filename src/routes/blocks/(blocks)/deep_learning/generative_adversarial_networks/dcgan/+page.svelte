<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";
  import results from "./results.png";

  const references = [
    {
      author: "A. Radford, L. Metz, and S. Chintala",
      title:
        "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks",
      year: "2016",
    },
  ];
</script>

<svelte:head>
  <title>DCGAN - World4AI</title>
  <meta
    name="description"
    content="DCGANs (deep convolutional generative adversarial networks) improved the performance of plain vanilla GANs by replacing all fully connected layers by convolutional neural networks."
  />
</svelte:head>

<Container>
  <h1>DCGAN</h1>
  <div class="separator" />
  <p>
    The fully connected GAN that we implemented in the last section generated
    MNIST images of subpar quality. This is to be expected, because plain
    vanilla fully connected neural networks are not well suited for images.
    Instead we usually utilize convolutional neural networks, if we want to
    achieve good performance in the area of computer vision. <Highlight
      >DCGANs</Highlight
    ><InternalLink id={1} type="reference" /> (deep convolutional generative adversarial
    networks) do just that by using CNNs for the generator and discriminator.
  </p>
  <p>
    The generator takes a latent vector as input. We follow the original
    implementation and use a noise of size 100. This noise is interpreted as 100
    feature maps of size 1x1, so each input is of size 100x1x1. The noise is
    processed by layers of transposed convolutions, batch norm and rectified
    linear units. Transposed convolutions upscale the images and reduce the
    number of channels, until we end up with images of size 1x64x64. The
    dimensionality of 64 was picked for convenience in order to follow the
    original parameters from the paper. We will once again work with MNIST, but
    we upscale the images to 64x64.
  </p>
  <PythonCode
    code={`class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=LATENT_SIZE, out_channels=1024, kernel_size=4, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=1,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=True,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)`}
  />
  <p>
    The discriminator on the other hand takes images as input and applies layers
    of convolutions. This procedure increases the number of channels and
    downscales the images. The last layer reduces the number of channels to 1
    and ends up with images of size 1x1x1. This value is flattened and is used
    as input for sigmoid in order to be interpreted as probability of being a
    real image.
  </p>
  <PythonCode
    code={`class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=True,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=4,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=4,
                padding=0,
                stride=1,
                bias=True,
            ),
            nn.Flatten(),
        )
`}
  />
  <p>
    When we train a DCGAN for 20 epochs we end with the following results. The
    images are more realistic and there are fewer artifacts compared to those
    from the last section.
  </p>
  <div class="flex justify-center ">
    <img src={results} alt="MNIST generated by a DCGAN" class="w-72" />
  </div>
</Container>
<Footer {references} />
