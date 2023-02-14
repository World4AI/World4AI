<script>
  import Table from "$lib/Table.svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Diagram from "$lib/diagram/Diagram.svelte";

  // basic block
  let gap = 1;
  let basicWidth = 90;
  let basicHeight = 100;
  let basicBoxWidth = 80;
  let basicBoxHeight = 20;
  let basicMaxWidth = "200px";

  const basicComponents = [
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "Conv2d: 3x3, S:1, P:1",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight / 2,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "BatchNorm2d",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicBoxHeight / 2 + gap,
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
          y: basicHeight - basicBoxHeight - gap,
        },
        {
          x: basicWidth / 2,
          y: basicHeight / 2 + basicBoxHeight / 2 + 3,
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
          y: basicHeight / 2 - basicBoxHeight / 2,
        },
        {
          x: basicWidth / 2,
          y: basicBoxHeight + 4,
        },
      ],
    },
  ];

  let header = ["Type", "Input Size", "Output Size"];
  let data = [
    ["Basic Block", "224x224x3", "224x224x64"],
    ["Basic Block", "224x224x64", "224x224x64"],
    ["Max Pooling", "224x224x64", "112x112x64"],
    ["Basic Block", "112x112x64", "112x112x128"],
    ["Basic Block", "112x112x128", "112x112x128"],
    ["Max Pooling", "112x112x128", "56x56x128"],
    ["Basic Block", "56x56x128", "56x56x256"],
    ["Basic Block", "56x56x256", "56x56x256"],
    ["Basic Block", "56x56x256", "56x56x256"],
    ["Max Pooling", "56x56x256", "28x28x256"],
    ["Basic Block", "28x28x256", "28x28x512"],
    ["Basic Block", "28x28x512", "28x28x512"],
    ["Basic Block", "28x28x512", "28x28x512"],
    ["Max Pooling", "28x28x512", "14x14x512"],
    ["Basic Block", "14x14x512", "14x14x512"],
    ["Basic Block", "14x14x512", "14x14x512"],
    ["Basic Block", "14x14x512", "14x14x512"],
    ["Max Pooling", "14x14x512", "7x7x512"],
    ["Dropout", "-", "-"],
    ["Fully Connected", "25088", "4096"],
    ["ReLU", "-", "-"],
    ["Dropout", "-", "-"],
    ["Fully Connected", "4096", "4096"],
    ["ReLU", "-", "-"],
    ["Fully Connected", "4096", "1000"],
    ["Softmax", "-", "-"],
  ];

  let references = [
    {
      author: "Simonyan, K., & Zisserman, A.",
      title:
        "Very deep convolutional networks for large-scale image recognition",
      journal: "",
      year: "2014",
      pages: "",
      volume: "",
      issue: "",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | VGG</title>
  <meta
    name="description"
    content="VGG is at heart a very simple convolutional neural network architecture. It stacks layers of convolutions followed by max pooling. But compared to AlexNet or LeNet-5 this architecture showed that deeper and deeper networks might be necessary to achieve truly impressive results."
  />
</svelte:head>

<h1>VGG</h1>
<div class="separator" />
<Container>
  <p>
    The VGG<InternalLink type={"reference"} id={1} /> architecture came from the
    visual geometry group, a computer vision research lab at Oxford university. The
    neural network is similar in spirit to LeNet-5 and AlexNet, where convolutional
    layers are stacked upon each other followed by a pooling layer, but vgg does
    so with many more layers and many more filters per layer. VGG also introduced
    a practice that is very common to this day. Unlike AlexNet, VGG does not apply
    any large filters, but uses only small patches of 3x3. Most modern convolutional
    networks use only 2x2 or 3x3 filters and VGG was the first network to introduce
    the practice. This design choice lead to the second place in the 2014 ImageNet
    object detection challenge.
  </p>
  <p>
    The VGG paper discussed networks of varying depth, from 11 layers to 19
    layers. We are going to discuss the 16 layer architecture, the so called
    VGG16 (architecture D in the paper).
  </p>

  <p>
    One of the greatest advantages of VGG is its repeatablity of calculations.
    In the 19 layer VGG network, a convolutional operation with the same kernel,
    stride and padding is repeated 2 to 3 times using the same steps. Only the
    amount of filters varies. That allows us to create a "basic block", that
    stacks a convolution operation, a batch normalization layer and the ReLU
    activation function. We can reuse that block over and over again. Be aware,
    that the BatchNorm2d layer was not used in the original VGG paper, but if
    you omit normalization step, the network will suffer from vanishing
    gradients.
  </p>
  <Diagram
    width={basicWidth}
    height={basicHeight}
    maxWidth={basicMaxWidth}
    components={basicComponents}
  />
  <p>
    The full VGG16 implementation looks as follows. All pooling layers have a
    kernel size and stride of 2.
  </p>
  <Table {header} {data} />
  <p>
    The original VGG16 architecture has 138 million of trainable parameters.
    Unless you have a powerful modern graphics card with a lot of vram, your
    program will crash. Our PyTorch implementation below runs without problems
    on the free version of Google Colab, but we had to use a batch size that is
    much smaller than the value provided in the paper, in order to fit the model
    and the data into the memory of the graphics card.
  </p>
</Container>

<Footer {references} />
