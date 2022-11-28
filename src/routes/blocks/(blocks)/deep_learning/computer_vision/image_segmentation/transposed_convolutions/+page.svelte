<script>
  import Hierarchy from "../../image_classification/_convolution/Hierarchy.svelte";
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Transposed from "../_convolution/Transposed.svelte";

  export let layers1 = [
    { width: 10, height: 10, channels: 3 },
    { width: 8, height: 8, channels: 8 },
    { width: 6, height: 6, channels: 16 },
    { width: 4, height: 4, channels: 32 },
    { width: 1, height: 1, channels: 64 },
  ];
  export let layers2 = [
    { width: 10, height: 10, channels: 3 },
    { width: 6, height: 6, channels: 16 },
    { width: 4, height: 4, channels: 32 },
    { width: 1, height: 1, channels: 64 },
    { width: 4, height: 4, channels: 32 },
    { width: 6, height: 6, channels: 16 },
    { width: 10, height: 10, channels: 3 },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Transposed Convolutions</title>
  <meta
    name="description"
    content="Often in computer vision we need to upsample images. Transposed convolution allow us to take a lower resolution shape and to produce a higher resolution shape."
  />
</svelte:head>

<h1>Transposed Convolutions</h1>
<div class="separator" />
<Container>
  <p>When we use convolutional neural networks, we usually go from a high resolution image to a low resolution image. This is called <Highlight>downsampling</Highlight>. Downsampling allows the receptive field to see larger and larger chunks of the original image and to output a feature vector that can be used for image classification.</p>
</Container>
<Container maxWidth={1200}>
  <Hierarchy maxWidth={1200} layers={layers1} />
</Container>

<Container>
  <p>But what if our desired output is a different image and not a vector? In image segmentation for example the desired mask has the same resolution as the input image.</p>
</Container>
<Container maxWidth={1200}>
  <Hierarchy maxWidth={1200} blockSize={15} gap={5} layerDistance={220} layers={layers2} />
</Container>
<Container>
  <p>We still need to downsample the image in the first part of the architecture in order to calculate the required features, but afterwards we need to find a way to <Highlight>upsample</Highlight> the image for the required task. </p>
  <p>So far we have utilized two strategies for downsampling: pooling layers and convolutions with a stride larger than 1. For upsampling we are going to use a techinque called <Highlight>transposed convolutions</Highlight>.</p>
  <p>In a normal convolution we take a patch of pixels or neurons from the previous layer and multiply that patch by the kernel. We sum the result, so that for each patch we get one single output neuron. In a transposed convolution we take a single input and multiply that input by the kernel, the output is a whole patch of neurons for each of the input neurons.</p>
  <p>In the example below we have a 2x2 input, a 3x3 kernel and we use a stride of 2. The stride is applied not to the input but to the output. We use the stride to place the generated paches of neurons. We multiply each of the four input cells by the kernel and thus have alltogether 4 3x3 patches. We place each of the patches moving with a stride of 2. You can simulate this behaviour by clicking on each of the input cells. You have probably noticed, that there is some overlap in the output. If that happens we simply add up all the overlapping results.</p>
  <Transposed />
  <p>If you still have trouble imagining the shapes of the input and the output, try to swap the input and the output. If you apply the standard convolution using the 3x3 kernel and the stride of 2, you will end up with a 2x2 feature map.</p>
  <div class="separator" />
</Container>
