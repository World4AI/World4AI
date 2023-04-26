<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Plus from "$lib/diagram/Plus.svelte";

  import results from "./generated_images.png";

  const references = [
    {
      author: "Oord, Aaron and Kalchbrenner, Nal and Kavukcuoglu, Koray",
      title: "Pixel Recurrent Neural Networks",
      year: "2016",
    },
  ];

  const imageLength = 4;
  const pixelSize = 25;
  const padding = 1;
  const kernel = 3;

  let activeRow = 0;
  let activeCol = 0;

  function f() {
    if (activeCol < 3) {
      activeCol += 1;
    } else if (activeCol === 3 && activeRow < 3) {
      activeCol = 0;
      activeRow += 1;
    } else {
      activeCol = activeRow = 0;
    }
  }
</script>

<svelte:head>
  <title>PixelCNN - World4AI</title>
  <meta
    name="description"
    content="PixelCNN is an anotregressive generative image model, that was developed by DeepMind. The model uses masked convolutions, thereby filtering out future pixels that the model should not have access to. At the time of the publishing the research at DeepMind produced state of the art results in image generation."
  />
</svelte:head>

<Container>
  <h1>PixelCNN</h1>
  <div class="separator" />
  <p>
    <Highlight>PixelCNN</Highlight><InternalLink id="1" type="reference" /> is an
    autoregressive generative image model that came out of DeepMind. The authors
    introduced 4 types of models simultaneously: RowLSTM, Diagonal BiLSTM, Multi-Scale
    PixelRNN and PixelCNN. While the models that were based on recurrent neural networks
    performed better originally, PixelCNN was a lot easier to parallelize. Over the
    next years a lot of improvement were done to PixelCNN, which resulted in a much
    better performance. In this section we will focus on the original PixelCNN implementation
    and look at some improvements in the coming sections.
  </p>
  <p>
    Our PyTorch implementation is inspired by <a
      href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html"
      target="_blank"
      rel="noreferrer">UvA Deep Learning Tutorials</a
    >. We recommend you check out their tutotorial if you want to get a deeper
    understanding of PixelCNN and Gated PixelCNN (will be covered in the next
    section).
  </p>
  <div class="separator" />

  <h2>Masked Convolutions</h2>
  <p>
    Let's utilize a stylized example of a 4x4 image in order to understand what
    role <Highlight>masked convolutions</Highlight> play and why they are necessary.
  </p>

  <SvgContainer maxWidth="120px">
    <svg viewBox="0 0 120 120">
      {#each Array(imageLength) as _, colIdx}
        {#each Array(imageLength) as _, rowIdx}
          <rect
            x={2 + colIdx * (pixelSize + 5)}
            y={2 + rowIdx * (pixelSize + 5)}
            width={pixelSize}
            height={pixelSize}
            class="fill-slate-300"
            stroke="black"
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    We need to process the image in such a way, that the size (height and width)
    of the input and the output feature maps are identical. So given a 4x4 image
    and a 3x3 convolution, we need a padding of 1 on each side of the 2d image.
  </p>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding * 2) as _, colIdx}
        {#each Array(imageLength + padding * 2) as _, rowIdx}
          <rect
            x={2 + colIdx * (pixelSize + 5)}
            y={2 + rowIdx * (pixelSize + 5)}
            width={pixelSize}
            height={pixelSize}
            class={colIdx < padding ||
            rowIdx < padding ||
            colIdx >= imageLength + padding ||
            rowIdx >= imageLength + padding
              ? "fill-white"
              : "fill-slate-300"}
            stroke="black"
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    This is required, because we want to end up with a probability distribution
    for each pixel. For each of the 4x4 pixels we end up with a 256-way softmax
    layer. Each of the 256 numbers represents the probability of the pixel to be
    one of the 256 integer values between 0 and 255. We can use those
    probabilities to sample pixel values from the multinomial distribution. So
    in our example we would start with a greyscale image of shape 1x4x4 and end
    up with 256x1x4x4.
  </p>
  <p>
    Now if you look at the kernel below, as indicated by the red dots, you will
    hopefully see a problem in this type of calculation. The very first output
    contains knowledge about future pixels. Autoregressive generative models
    take in previous pixels to generate future pixels. If previous pixels
    contain knowledge about the future, that would be considered cheating and
    while our training process would look great, inference would produce
    garbage, because the model would not have learned how to generate pixels
    based solely on the past.
  </p>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding * 2) as _, colIdx}
        {#each Array(imageLength + padding * 2) as _, rowIdx}
          <rect
            x={2 + colIdx * (pixelSize + 5)}
            y={2 + rowIdx * (pixelSize + 5)}
            width={pixelSize}
            height={pixelSize}
            class={colIdx < padding ||
            rowIdx < padding ||
            colIdx >= imageLength + padding ||
            rowIdx >= imageLength + padding
              ? "fill-white"
              : "fill-slate-300"}
            stroke="black"
          />
        {/each}
      {/each}
      {#each Array(kernel) as _, colIdx}
        {#each Array(kernel) as _, rowIdx}
          <circle
            cx={2 + colIdx * (pixelSize + 5) + pixelSize / 2}
            cy={2 + rowIdx * (pixelSize + 5) + pixelSize / 2}
            r={4}
            class="fill-red-400"
            stroke="black"
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    To deal with this problem, we need to apply a mask to the kernel.
    Essentially we multiply the kernel values that would access future pixels
    with zeros, thereby zeroing out the weights of kernel positions that relate
    to future pixels.
  </p>
  <ButtonContainer>
    <PlayButton {f} delta={500} />
  </ButtonContainer>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding * 2) as _, colIdx}
        {#each Array(imageLength + padding * 2) as _, rowIdx}
          <rect
            x={2 + colIdx * (pixelSize + 5)}
            y={2 + rowIdx * (pixelSize + 5)}
            width={pixelSize}
            height={pixelSize}
            class={colIdx < padding ||
            rowIdx < padding ||
            colIdx >= imageLength + padding ||
            rowIdx >= imageLength + padding
              ? "fill-white"
              : "fill-slate-300"}
            stroke="black"
          />
        {/each}
      {/each}
      <!-- draw kernel -->
      {#each Array(kernel) as _, colIdx}
        {#each Array(kernel) as _, rowIdx}
          <circle
            cx={2 + (colIdx + activeCol) * (pixelSize + 5) + pixelSize / 2}
            cy={2 + (rowIdx + activeRow) * (pixelSize + 5) + pixelSize / 2}
            r={4}
            class={rowIdx < padding || (rowIdx === padding && colIdx < padding)
              ? "fill-red-400"
              : "fill-none"}
            stroke="black"
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    There are two types of kernel masks that are used for PixelRNNs. The above
    mask is of type <Highlight>A</Highlight>. A type A mask zeroes out all
    values up to the position we would like to produceIn the below example on
    the other hand we use a mask of type <Highlight>B</Highlight>. A type B mask
    does not zero out the position in the input feature map that corresponds to
    the position in the output feature map.
  </p>
  <ButtonContainer>
    <PlayButton {f} delta={500} />
  </ButtonContainer>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding * 2) as _, colIdx}
        {#each Array(imageLength + padding * 2) as _, rowIdx}
          <rect
            x={2 + colIdx * (pixelSize + 5)}
            y={2 + rowIdx * (pixelSize + 5)}
            width={pixelSize}
            height={pixelSize}
            class={colIdx < padding ||
            rowIdx < padding ||
            colIdx >= imageLength + padding ||
            rowIdx >= imageLength + padding
              ? "fill-white"
              : "fill-slate-300"}
            stroke="black"
          />
        {/each}
      {/each}
      <!-- draw kernel -->
      {#each Array(kernel) as _, colIdx}
        {#each Array(kernel) as _, rowIdx}
          <circle
            cx={2 + (colIdx + activeCol) * (pixelSize + 5) + pixelSize / 2}
            cy={2 + (rowIdx + activeRow) * (pixelSize + 5) + pixelSize / 2}
            r={4}
            class={rowIdx < padding || (rowIdx === padding && colIdx <= padding)
              ? "fill-red-400"
              : "fill-none"}
            stroke="black"
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    The PixelCNN architecture uses type A masks for the input image, while type
    B masks are applied to all intermediary results. When we get the actual
    image as input and try to generate a pixel, we have to use mask A in order
    to hide the actual pixel the model tries to predict. So when we try to
    predict the very first pixel, the model can only look at zero padded values.
    After the first processing step with mask of type A the values do not
    contain actual information about the original pixel in that position, but
    only information about pixel values that surround the pixel we are trying to
    produce, so we can use mask B safely.
  </p>
  <p>
    We create a special <code>MaskedConvolution</code> module, that we can reuse
    in several places. The module implements a different mask depending on the type
    of the mask. The padding of the convolution operation is determined automatically
    based on the kernel size and dilations. Dilations are not explicitly mentioned
    in the research paper, but they seem to slightly imrove the quality of the generated
    images.
  </p>
  <PythonCode
    code={`class MaskedConvolution(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), mask_type="B", dilation=1
    ):
        super().__init__()

        assert mask_type in ["A", "B"]
        mask = torch.zeros(kernel_size)
        mask[: kernel_size[0] // 2, :] = 1
        if mask_type == "A":
            mask[kernel_size[0] // 2, : kernel_size[1] // 2] = 1
        elif mask_type == "B":
            mask[kernel_size[0] // 2, : kernel_size[1] // 2 + 1] = 1
        self.register_buffer("mask", mask)

        # add conv2d layer
        padding = tuple([dilation * (size - 1) // 2 for size in kernel_size])
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

    def forward(self, x):
        with torch.inference_mode():
            self.conv.weight *= self.mask
        return self.conv(x)
`}
  />
  <div class="separator" />

  <h2>Skip Connections</h2>
  <p>
    In order to facilitate training, PixelCNN utilizes skip connections, by
    constructing residual blocks. The residual block scales down the number of
    hidden features from 2h to h, before a masked 'B' convolution is applied.
    Afterwards the dimension is scaled up again and the original input to the
    block and the output are summed.
  </p>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 220 220">
      <g transform="translate(-5 -10)">
        <Plus x={30} y={40} radius={10} offset={4} />
        <Block
          x="160"
          y="200"
          width="100"
          height="30"
          text="1x1 Conv"
          fontSize="20px"
          class="fill-blue-100"
        />
        <Block
          x="160"
          y="120"
          width="100"
          height="30"
          text="3x3 Conv"
          fontSize="20px"
          class="fill-red-100"
        />
        <Block
          x="160"
          y="40"
          width="100"
          height="30"
          text="1x1 Conv"
          fontSize="20px"
          class="fill-blue-100"
        />
        <Arrow
          data={[
            { x: 30, y: 220 },
            { x: 30, y: 60 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="4 4"
          moving={true}
          speed="80"
        />
        <Arrow
          data={[
            { x: 30, y: 200 },
            { x: 100, y: 200 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="4 4"
          moving={true}
          speed="80"
        />
        <Arrow
          data={[
            { x: 160, y: 180 },
            { x: 160, y: 145 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="4 4"
          moving={true}
          speed="80"
        />
        <Arrow
          data={[
            { x: 160, y: 100 },
            { x: 160, y: 65 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="4 4"
          moving={true}
          speed="80"
        />
        <Arrow
          data={[
            { x: 110, y: 40 },
            { x: 50, y: 40 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="4 4"
          moving={true}
          speed="80"
        />
        <Block
          x="70"
          y="180"
          width="25"
          height="25"
          text="2h"
          fontSize="15px"
        />
        <Block x="70" y="60" width="25" height="25" text="2h" fontSize="15px" />
        <Block x="140" y="80" width="25" height="25" text="h" fontSize="15px" />
        <Block
          x="140"
          y="160"
          width="25"
          height="25"
          text="h"
          fontSize="15px"
        />
      </g>
    </svg>
  </SvgContainer>
  <p>
    Additionally to the behaviour described above, we add <code
      >BatchNorm2d</code
    > before each activation function. This helps out with the training stability
    and overfitting.
  </p>
  <PythonCode
    code={`class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            MaskedConvolution(
                in_channels=hidden_dim, out_channels=hidden_dim, dilation=dilation
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=1
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
        )

    def forward(self, x):
        block = self.block(x)
        return x + block`}
  />
  <div class="separator" />

  <h2>Model</h2>
  <p>
    At this point we have all the ingredients to describe a PixelCNN
    architecture. We use a 7x7 masked convolutional layer of type 'A' to the
    input image. The output is followed up by 7 residual blocks, which utilize
    masked convolutions of type 'B'. The final outputs adjust the number of
    feature maps to the desired 256 and the softmax nonlinearity is applied.
  </p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 220 200">
      <Arrow
        data={[
          { x: 140, y: 0 },
          { x: 140, y: 190 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
        moving={true}
        speed="80"
      />
      <Block
        x="140"
        y="40"
        width="150"
        height="30"
        text="7x7 Conv, Type 'A'"
        fontSize="15px"
        class="fill-blue-200"
      />
      <Block
        x="140"
        y="80"
        width="150"
        height="30"
        text="Skip Connections"
        fontSize="15px"
        class="fill-blue-200"
      />
      <Block x="20" y="80" width="25" height="25" text="7x" fontSize="15px" />
      <Block
        x="140"
        y="120"
        width="150"
        height="30"
        text="1x1 Conv"
        fontSize="15px"
        class="fill-blue-200"
      />
      <Block x="20" y="120" width="25" height="25" text="2x" fontSize="15px" />
      <Block
        x="140"
        y="160"
        width="150"
        height="30"
        text="256-way Softmax"
        class="fill-blue-200"
        fontSize="15px"
      />
    </svg>
  </SvgContainer>
  <p>
    We add different types of dialitions for each ResidialBlock to allow the
    model to attend to pixels, that are farther away from the current position.
    You can experiment with those values to see if you can achieve better
    results.
  </p>
  <PythonCode
    code={`class PixelCNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            MaskedConvolution(
                in_channels=1,
                out_channels=hidden_dim * 2,
                kernel_size=(7, 7),
                mask_type="A",
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dilation=1),
            ResidualBlock(hidden_dim, dilation=2),
            ResidualBlock(hidden_dim, dilation=1),
            ResidualBlock(hidden_dim, dilation=3),
            ResidualBlock(hidden_dim, dilation=1),
            ResidualBlock(hidden_dim, dilation=2),
            ResidualBlock(hidden_dim, dilation=1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, 256, kernel_size=1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 256, 1, 28, 28)
        return x`}
  />
  <p>
    The qualityt of the generated images is far from optimal. Some of the images
    actually correspond to digits, others look 'digit-like', but overall there
    is room for improvement. Gated PixelCNNs will allow us to generate higher
    quality images. This is going to be the topic of the next section.
  </p>
  <div class="flex justify-center items-center">
    <img src={results} alt="Generated MNIST Images" class="w-96" />
  </div>
</Container>
<Footer {references} />
