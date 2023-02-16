<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";

  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Circle from "$lib/diagram/Circle.svelte";
</script>

<svelte:head>
  <title>Computer Vision - World4AI</title>
  <meta
    name="description"
    content="Computer vision is strongly inspired by human vision. It encompasses a large variety of tasks, but the main categories could be subdivided into image recognition, object detection and semantic segmentation."
  />
</svelte:head>

<h1>Computer Vision</h1>
<div class="separator" />

<Container>
  <p>
    <Highlight>Computer vision</Highlight> encompasses quite a large variety of tasks
    but in a nutshell we could say that computer vision tasks try to emulate animal
    vision. The computer receives images or videos as input and and is expected to
    processes those in a useful manner.
  </p>
  <p>
    Broadly speaking many computer vision tasks can be subdivided into 3 major
    categories. Let's use a very simple dummy example in order to get an
    intuition for what those categories are: an image of a cirlce.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 200 200" class="border">
      <Circle x={100} y={100} r={25} color="none" />
    </svg>
  </SvgContainer>
  <p>
    The goal of an <Highlight>image classification</Highlight> task would be to assign
    the correct label to an image. The model returns the probabilities for different
    labels and the label with the highest probability wins out. This is what we have
    been doing with the MNIST dataset so far.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 200 200" class="border">
      <Block x={27} y={16} width={50} height={25} text="Circle" fontSize={14} />
      <Circle x={100} y={100} r={25} color="none" />
    </svg>
  </SvgContainer>
  <p>
    <Highlight>Object detection</Highlight> models find certain objects in an image
    and draw a bounding box around that object.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 200 200" class="border">
      <Block x={100} y={100} width={58} height={58} />
      <Circle x={100} y={100} r={25} color="none" />
    </svg>
  </SvgContainer>
  <p>
    Semantic segmentation models assign each pixel in the image a certain
    category. Below for example we segment the image into the circle and the
    background.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 200 200" class="border bg-slate-300">
      <Circle x={100} y={100} r={25} color="var(--main-color-1)" />
    </svg>
  </SvgContainer>
  <p>
    All of above mentioned tasks are different in some regards, but usually they
    all utilize so called <Highlight>convolutional neural networks</Highlight>.
    We are going to cover those in the upcoming section.
  </p>
  <div class="separator" />
</Container>
