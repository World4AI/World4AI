<script>
  import Table from "$lib/Table.svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Convolution from "../../computer_vision/_convolution/Convolution.svelte";

  let header = ["Type", "Repeat", "Parameters"];
  let data = [
    ["Convolution 2D", "", "7x7x64"],
    ["BatchNorm2D", "", ""],
    ["ReLU", "", ""],
    ["Max Pooling", "", "Filter: 3x3, Stride: 2"],
    ["Basic Block", "3", "3x3x64"],
    ["Basic Block", "4", "3x3x128"],
    ["Basic Block", "6", "3x3x256"],
    ["Basic Block", "3", "3x3x512"],
    ["Adaptive Avg. Pooling", "", "512"],
    ["Fully Connected", "", "1000"],
    ["Softmax", "", "1000"],
  ];

  const references = [
    {
      author: "K. He, X. Zhang, S. Ren and J. Sun",
      title: "Deep Residual Learning for Image Recognition",
      journal:
        "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      year: "2016",
      pages: "770-778",
      volume: "",
      issue: "",
    },
  ];

  let gap = 1;

  // basic block
  let basicWidth = 100;
  let basicHeight = 300;
  let basicBoxWidth = 60;
  let basicBoxHeight = 20;
  let basicMaxWidth = "220px";

  const numComponents = 9;
  const vertGap =
    (basicHeight - numComponents - 1 * basicBoxHeight) / (numComponents - 2);

  const basicComponents = [
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "Input",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap - vertGap,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "Conv2d",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap - vertGap * 2,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "BatchNorm2d",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap - vertGap * 3,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "ReLU",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap - vertGap * 4,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "Conv2d",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap - vertGap * 5,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "BatchNorm",
      color: "var(--main-color-4)",
    },
    {
      type: "plus",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap - vertGap * 6,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "BatchNorm",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap - vertGap * 7,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "ReLU",
      color: "var(--main-color-4)",
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight,
        },
        {
          x: basicWidth / 2,
          y: basicHeight - vertGap + 3,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight - vertGap,
        },
        {
          x: basicWidth / 2,
          y: basicHeight - vertGap * 2 + 3,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight - vertGap * 2,
        },
        {
          x: basicWidth / 2,
          y: basicHeight - vertGap * 3 + 3,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight - vertGap * 3,
        },
        {
          x: basicWidth / 2,
          y: basicHeight - vertGap * 4 + 3,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight - vertGap * 4,
        },
        {
          x: basicWidth / 2,
          y: basicHeight - vertGap * 5 + 3,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight - vertGap * 5,
        },
        {
          x: basicWidth / 2,
          y: basicHeight - vertGap * 6,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight - vertGap * 6,
        },
        {
          x: basicWidth / 2,
          y: basicHeight - vertGap * 7 + 3,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2 - basicBoxWidth / 2,
          y: basicHeight - basicBoxHeight + basicBoxHeight / 2,
        },
        {
          x: basicWidth / 2 - basicBoxWidth / 2 - 15,
          y: basicHeight - basicBoxHeight + basicBoxHeight / 2,
        },
        {
          x: basicWidth / 2 - basicBoxWidth / 2 - 15,
          y: basicHeight - basicBoxHeight + basicBoxHeight / 2 - vertGap * 6,
        },
        {
          x: basicWidth / 2 - 10,
          y: basicHeight - basicBoxHeight + basicBoxHeight / 2 - vertGap * 6,
        },
      ],
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | ResNet</title>
  <meta
    name="description"
    content="The ResNet convolutional neural network architecture introduced skip connections, which allowed to established a network with 152 layers, which won the 2015 ImageNet classification competition."
  />
</svelte:head>

<h1>ResNet</h1>
<div class="separator" />
<Container>
  <p>
    We have introduced and discussed <a
      href="/blocks/deep_learning/stability_speedup/skip_connections"
      >skip connections</a
    >
    in a previous chapter. This time around we will discuss the ResNet <InternalLink
      type={"reference"}
      id={1}
    /> architecture, which introduced skip connections to the world.
  </p>
  <p>
    Several ResNet variants were introduced in the paper. From ResNet18 with
    just 18 layers all the way to ResNet152, with 152 layers. The 152 layer
    variant won the ILSVRC15 classification challenge with a 3.57 top-5 error
    rate. Remember that just the year before GoogLeNet achieved 6.67.
  </p>
  <p>
    In this section we will focus on the ResNet34 architecture. But if you are
    interested in implementing the 152 layer architecture, you should be able to
    extend the code below without many difficulties.
  </p>
  <p>
    Similar to the architectures we studied before, ResNet34 is based on many
    basic building blocks. Only this time the block is based on skip
    connections.
  </p>
  <!--
  <Diagram
    width={basicWidth}
    height={basicHeight}
    maxWidth={basicMaxWidth}
    components={basicComponents}
  />
  -->
  <p>
    The block consists of two convolutions. The skip connection goes directly
    from the output of the previous layer, past the two convolutions and is
    added to the usual path, before the ReLU is applied to the sum. Bear in
    mind, that this block is slightly different for larger ResNet architectures.
  </p>
  <p>
    Usually the convolutions have a kernel size of 3x3 with a padding of 1, a
    step size of 1. Additionally the number of filters usually stays constant.
    This makes the output size equal to the input size and we do not have any
    trouble adding the input to the output of the second convolution.
  </p>
  <Convolution
    kernel={3}
    stride={1}
    padding={1}
    imageWidth={6}
    imageHeight={6}
    showOutput={true}
  />
  <p>
    Yet sometimes we reduce the resolution by 2 using a stride of 2 and increase
    the number of filters by two. If we have a 100x100x3 image, we would end up
    with a 50x50x6 image. This procedure keeps the number of paramters constant,
    yet we can not apply the addition, because the dimensionality of the input
    and the output differ.
  </p>
  <Convolution
    kernel={3}
    stride={2}
    padding={1}
    imageWidth={6}
    imageHeight={6}
    showOutput={true}
  />
  <p>
    For that purpose the input is also downsampled using a kernel size of 1 and
    a stride of 2, while the number of channels is doubled.
  </p>
  <Convolution
    maxWidth={350}
    kernel={1}
    stride={2}
    padding={0}
    imageWidth={6}
    imageHeight={6}
    showOutput={true}
  />
  <p>
    This ResNet architecture uses alsmost no pooling layers for downsampling,
    instead if the need arises to downsample, the first convolution in the basic
    block uses a stride of 2.
  </p>
  <p>
    The overall architecture looks as below. The same building blocks are
    repeated over and over again, while halving the resolution and doubling the
    number of channels from time to time.
  </p>
  <Table {header} {data} />
  <div class="separator" />
</Container>

<Footer {references} />
