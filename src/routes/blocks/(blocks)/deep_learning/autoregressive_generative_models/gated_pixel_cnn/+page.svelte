<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  import Block from "$lib/diagram/Block.svelte";
  import Plus from "$lib/diagram/Plus.svelte";
  import Multiply from "$lib/diagram/Multiply.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  import results from "./generated_images.png";

  const references = [
    {
      author:
        " van den Oord, Aäron and Kalchbrenner, Nal and Vinyals, Oriol and Espeholt, Lasse and Graves, Alex and Kavukcuoglu, Koray",
      title: "Conditional Image Generation with PixelCNN Decoders",
      year: "2016",
    },
  ];

  const centerIdx = 4;
  const numPixels = 9;
  const pixelSize = 22;
  const gap = 5;
  let layerCommon = 0;
  let layerMasked = 0;
  let layerVertical = 0;
  let layerHorizontal = 0;

  function fCommon() {
    if (layerCommon === 4) {
      layerCommon = 0;
    } else {
      layerCommon += 1;
    }
  }

  function fMasked() {
    if (layerMasked === 4) {
      layerMasked = 0;
    } else {
      layerMasked += 1;
    }
  }

  function fVertical() {
    if (layerVertical === 4) {
      layerVertical = 0;
    } else {
      layerVertical += 1;
    }
  }

  function fHorizontal() {
    if (layerHorizontal === 4) {
      layerHorizontal = 0;
    } else {
      layerHorizontal += 1;
    }
  }

  //assume padding of 1 and kernel size 3
  function colorCommon(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx - centerIdx);
    let colDistance = Math.abs(colIdx - centerIdx);
    if (colIdx === centerIdx && rowIdx === centerIdx) {
      return "black";
    } else if (rowDistance <= layer && colDistance <= layer) {
      return `hsl(10, ${100 / Math.max(rowDistance, colDistance) + 10}%, 50%)`;
    } else {
      return "none";
    }
  }

  function colorMasked(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx - centerIdx);
    let colDistance = Math.abs(colIdx - centerIdx);
    let beyondMiddleLineRow = numPixels - colIdx;
    if (colIdx === centerIdx && rowIdx === centerIdx) {
      return "black";
    } else if (
      rowDistance <= layer &&
      colDistance <= layer &&
      rowIdx <= centerIdx &&
      rowIdx < beyondMiddleLineRow
    ) {
      return `hsl(10, ${100 / Math.max(rowDistance, colDistance) + 10}%, 50%)`;
    } else {
      return "none";
    }
  }

  //assume padding of 1 and kernel size 3
  function colorVertical(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx - centerIdx);
    let colDistance = Math.abs(colIdx - centerIdx);
    if (colIdx === centerIdx && rowIdx === centerIdx) {
      return "black";
    } else if (
      rowDistance <= layer &&
      colDistance <= layer &&
      rowIdx < centerIdx
    ) {
      return `hsl(10, ${100 / Math.max(rowDistance, colDistance) + 10}%, 50%)`;
    } else {
      return "none";
    }
  }

  function colorHorizontal(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx - centerIdx);
    let colDistance = Math.abs(colIdx - centerIdx);
    if (colIdx === centerIdx && rowIdx === centerIdx) {
      return "black";
    } else if (
      rowDistance <= layer &&
      colDistance <= layer &&
      colIdx < centerIdx &&
      rowIdx === centerIdx
    ) {
      return `hsl(10, ${100 / Math.max(rowDistance, colDistance) + 10}%, 50%)`;
    } else {
      return "none";
    }
  }
</script>

<svelte:head>
  <title>Gated PixelCNN - World4AI</title>
  <meta
    name="description"
    content="The gated PixelCNN model was developed by DeepMind to improve the generative quality of the common PixelCNN. The model utilizes two stacks of masked convolutions and a gated architecture to improve the performance."
  />
</svelte:head>

<Container>
  <h1>Gated PixelCNN</h1>
  <div class="separator" />
  <p>
    Shortly after the initial release of the PixelCNN architecture, DeepMind
    released the <Highlight>Gated PixelCNN</Highlight><InternalLink
      id={1}
      type="reference"
    />. The paper introduced several improvements simultaneously, that reduced
    the gap with the recurrent PixelRNN.
  </p>
  <div class="separator" />

  <h2>Vertical and horizontal Stacks</h2>
  <p>
    The PixelCNN has a limitation, that is not obvious at first glance. To
    explain that limitation let's remember how a convolutional neural network
    usually works. The very first layer applies convolutions to a tight
    receptive field around a particular pixel. If we apply a 3x3 convolution,
    then the neural network can only look at the immediate surroundings of a
    particular pixel. But as we stack more and more convolutional layers on top
    of each other, the receptive field starts to grow.
  </p>
  <p>
    In this interactive example we assume that all calculations are considered
    from the perspective of the black pixel, the kernel size is 3x3 and the
    padding is always 1 in order to keep the size of the image constant.
  </p>
  <ButtonContainer>
    <PlayButton f={fCommon} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect
            x={gap + colIdx * (pixelSize + gap)}
            y={gap + rowIdx * (pixelSize + gap)}
            width={pixelSize}
            height={pixelSize}
            class="stroke-black"
            fill={colorCommon(colIdx, rowIdx, layerCommon)}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>

  <p>
    Now let's see how the receptive field grows, once we incorporate masked
    convolutions.
  </p>
  <ButtonContainer>
    <PlayButton f={fMasked} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect
            x={gap + colIdx * (pixelSize + gap)}
            y={gap + rowIdx * (pixelSize + gap)}
            width={pixelSize}
            height={pixelSize}
            class="stroke-black"
            fill={colorMasked(colIdx, rowIdx, layerMasked)}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    While the receptive field grows, we are left with a blind spot. Many pixels
    above the black dot are not taken into the account, which will most likely
    deteriorate the performance.
  </p>
  <PythonCode
    code={`class MaskedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, mask, dilation=1):
        super().__init__()
        kernel_size = mask.shape
        padding = tuple([dilation * (size - 1) // 2 for size in kernel_size])

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.register_buffer("mask", mask)

    def forward(self, x):
        with torch.no_grad():
            self.conv.weight *= self.mask
        return self.conv(x)`}
  />
  <p>
    To deal with this problem the researchers at DeepMind separated the
    convolution into two distinct stacks: the <Highlight
      >vertical stack</Highlight
    >, which processes the pixels above the black pixel and the <Highlight
      >horizontal stack</Highlight
    >, which processes the pixels to the left.
  </p>
  <p>
    You can think about the vertical stack as a regular convolution, that can
    only access the upper half of the image.
  </p>
  <ButtonContainer>
    <PlayButton f={fVertical} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect
            x={gap + colIdx * (pixelSize + gap)}
            y={gap + rowIdx * (pixelSize + gap)}
            width={pixelSize}
            height={pixelSize}
            stroke="var(--text-color)"
            fill={colorVertical(colIdx, rowIdx, layerVertical)}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    The horizontal stack is a 1d convolution that processes the pixels to the
    left.
  </p>
  <ButtonContainer>
    <PlayButton f={fHorizontal} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect
            x={gap + colIdx * (pixelSize + gap)}
            y={gap + rowIdx * (pixelSize + gap)}
            width={pixelSize}
            height={pixelSize}
            stroke="var(--text-color)"
            fill={colorHorizontal(colIdx, rowIdx, layerHorizontal)}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>The combination of both produces the desired output.</p>
  <PythonCode
    code={`class VerticalStackConvolution(MaskedConvolution):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, mask_type="B", dilation=1
    ):
        assert mask_type in ["A", "B"]
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0
        if mask_type == "A":
            mask[kernel_size // 2, :] = 0

        super().__init__(in_channels, out_channels, mask, dilation=dilation)


class HorizontalStackConvolution(MaskedConvolution):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, mask_type="B", dilation=1
    ):
        assert mask_type in ["A", "B"]
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0
        if mask_type == "A":
            mask[0, kernel_size // 2] = 0
        super().__init__(in_channels, out_channels, mask, dilation=dilation)`}
  />
  <div class="separator" />

  <h2>Gated Architecture</h2>
  <p>
    The gated PixelCNN architecture was developed in order to close the
    performance gap between the PixelCNN and the RowLSTM. The researcher
    hypothesised, that the multiplicative units from an LSTM can help the model
    to learn more complex patterns and introduced similar units to the
    convolutional layers.
  </p>
  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 400 400" class="border">
      <!-- upper part -->
      <g>
        <Arrow
          data={[
            { x: 0, y: 70 },
            { x: 105, y: 70 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 120, y: 70 },
            { x: 120, y: 20 },
            { x: 155, y: 20 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 120, y: 70 },
            { x: 120, y: 120 },
            { x: 155, y: 120 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 220, y: 20 },
            { x: 260, y: 20 },
            { x: 260, y: 50 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 220, y: 120 },
            { x: 260, y: 120 },
            { x: 260, y: 90 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 260, y: 70 },
            { x: 390, y: 70 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Block
          x={40}
          y={70}
          width={55}
          height={25}
          text="n \times n"
          fontSize={15}
          type="latex"
          class="fill-green-100"
        />
        <Block x={120} y={70} width={15} height={15} class="fill-blue-100" />
        <Multiply x={260} y={70} radius={10} class="fill-red-300" />
        <Block
          x={190}
          y={20}
          width={55}
          height={25}
          text="tanh"
          fontSize={15}
          type="latex"
          class="fill-yellow-100"
        />
        <Block
          x={190}
          y={120}
          width={55}
          height={25}
          text="\sigma"
          fontSize={15}
          type="latex"
          class="fill-yellow-100"
        />
      </g>
      <!-- conection -->
      <Arrow
        data={[
          { x: 90, y: 70 },
          { x: 90, y: 250 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="6 6"
        moving={true}
      />

      <Block
        x={90}
        y={170}
        width={55}
        height={25}
        text="1 \times 1"
        fontSize={15}
        type="latex"
        class="fill-green-100"
      />

      <!-- lower part -->
      <g transform="translate(0 200)">
        <Arrow
          data={[
            { x: 0, y: 180 },
            { x: 390, y: 180 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 40, y: 180 },
            { x: 40, y: 90 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 0, y: 70 },
            { x: 105, y: 70 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 120, y: 70 },
            { x: 120, y: 20 },
            { x: 155, y: 20 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 260, y: 70 },
            { x: 350, y: 70 },
            { x: 350, y: 160 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 120, y: 70 },
            { x: 120, y: 120 },
            { x: 155, y: 120 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 220, y: 20 },
            { x: 260, y: 20 },
            { x: 260, y: 50 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Arrow
          data={[
            { x: 220, y: 120 },
            { x: 260, y: 120 },
            { x: 260, y: 90 },
          ]}
          strokeWidth={2}
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
        />
        <Block
          x={40}
          y={70}
          width={55}
          height={25}
          text="1 \times n"
          fontSize={15}
          type="latex"
          class="fill-green-100"
        />
        <Plus x={90} y={70} radius={10} offset={4} class="fill-red-300" />
        <Plus x={350} y={180} radius={10} offset={4} class="fill-red-300" />
        <Block
          x={350}
          y={70}
          width={55}
          height={25}
          text="1 \times 1"
          fontSize={15}
          type="latex"
          class="fill-green-100"
        />
        <Block x={120} y={70} width={15} height={15} class="fill-blue-100" />
        <Multiply x={260} y={70} radius={10} class="fill-red-300" />
        <Block
          x={190}
          y={20}
          width={55}
          height={25}
          text="tanh"
          fontSize={15}
          type="latex"
          class="fill-yellow-100"
        />
        <Block
          x={190}
          y={120}
          width={55}
          height={25}
          text="\sigma"
          fontSize={15}
          type="latex"
          class="fill-yellow-100"
        />
      </g>
    </svg>
  </SvgContainer>
  <p>
    Let's start our discussion with the upper part of the graph: the vertical
    stack. The vertical layer receives the output from the previous vertical
    stack and applies a <Latex>n \times n</Latex> masked convolution of type 'B',
    such that the mask only looks at the above pixels. The convolution takes in <Latex
      >p</Latex
    >
    feature maps and produces twice that amount as the output. This is done because
    one half goes into the <Latex>\tanh</Latex> and the other goes into the sigmoid
    activation <Latex>\sigma</Latex>. We multiply both results positionwise. In
    essence we can interpret the sigmoid output as a gate, that decides which
    part of the <Latex>\tanh</Latex> output is allowed to flow.
  </p>
  <p>
    The lower part of the graph is the horizontal stack. First we process the
    output from the vertical convolution through a 1x1 convolution and add that
    result to the output of the horizontal convolution. That way the model can
    attend to all above pixels and all pixels to the left. Second we use skip
    connections in the vertical stack in order to facilitate training.
  </p>
  <p>
    Lastly the PixelCNN paper focused on conditional models. For example we
    would like to condition the model on the label we would like to produce. As
    we are dealing with MNIST, we could use the numbers 0-9 as an additional
    input to the model, so that it can learn to generate specific numbers on
    demand. This should make it easier for a model to create coherent numbers.
  </p>
  <PythonCode
    code={`class ConditionalGatedResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.v = VerticalStackConvolution(
            in_channels,
            out_channels=2 * in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.h = HorizontalStackConvolution(
            in_channels,
            out_channels=2 * in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.v_to_h = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=1)
        self.v_to_res = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.v_embedding = nn.Embedding(num_embeddings=10, embedding_dim=in_channels)
        self.h_embedding = nn.Embedding(num_embeddings=10, embedding_dim=in_channels)

    def forward(self, v_prev, h_prev, num_cls):
        # calculate embeddings to condition the model
        v_embedding = self.v_embedding(num_cls).unsqueeze(-1).unsqueeze(-1)
        h_embedding = self.h_embedding(num_cls).unsqueeze(-1).unsqueeze(-1)

        # vertical stack
        v = self.v(v_prev + v_embedding)
        v_f, v_g = v.chunk(2, dim=1)
        v_out = torch.tanh(v_f) * torch.sigmoid(v_g)

        # vertical to horizontal
        v_to_h = self.v_to_h(v)

        # horizontal stack
        h = self.h(h_prev + h_embedding) + v_to_h
        h_f, h_g = h.chunk(2, dim=1)
        h_out = torch.tanh(h_f) * torch.sigmoid(h_g)

        # skip connection
        h_out = self.v_to_res(h_out)
        h_out += h_prev

        return v_out, h_out


class ConditionalGatedPixelCNN(nn.Module):
    def __init__(self, hidden_dim, dilations=[1, 2, 1, 4, 1, 2, 1]):
        super().__init__()
        self.v = VerticalStackConvolution(
            in_channels=1, out_channels=hidden_dim, kernel_size=7, mask_type="A"
        )
        self.h = HorizontalStackConvolution(
            in_channels=1, kernel_size=7, out_channels=hidden_dim, mask_type="A"
        )

        self.gated_residual_blocks = nn.ModuleList(
            [
                ConditionalGatedResidualBlock(
                    hidden_dim, kernel_size=3, dilation=dilation
                )
                for dilation in dilations
            ]
        )

        self.conv = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1
        )

        # we apply a 256 way softmax
        self.output = nn.Conv2d(in_channels=hidden_dim, out_channels=256, kernel_size=1)

    def forward(self, x, label):
        v = self.v(x)
        h = self.h(x)

        for gated_layer in self.gated_residual_blocks:
            v, h = gated_layer(v, h, label)

        out = self.conv(F.relu(h))
        out = self.output(F.relu(out))
        # from Batch, Classes, Height, Width to Batch, Classes, Channel, Height, Width
        out = out.unsqueeze(dim=2)
        return out`}
  />
  <p>
    If we train our model for 25 epochs we get images similar to those below.
    The quality of the generated images is clearly a lot better than those we
    created in the previous section.
  </p>
  <div class="flex justify-center items-center">
    <img src={results} alt="Generated MNIST Images" class="w-96" />
  </div>
  <div class="separator" />
</Container>

<Footer {references} />
