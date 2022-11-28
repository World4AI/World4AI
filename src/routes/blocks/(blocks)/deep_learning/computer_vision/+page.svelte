<script>
  import Container from "$lib/Container.svelte";
  import Alert from "$lib/Alert.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Computer Vision</title>
  <meta
    name="description"
    content="Computer vision deals with image recognition, object classification and object segmentation."
  />
</svelte:head>

<Alert type="danger">
  This chapter is work in progress
</Alert>
<h1>Computer Vision</h1>
<div class="separator" />
<Container>
  <p>
    Computer vision is a field that tries to imitate human-level abilities to
    perceive the world and to extract useful information from the receptive
    field. In its simplest form we feed images into a neural network and receive
    a predicted category as output. While we have already achieved relatively
    good performance using simple feed forward neural networks on the MNIST
    dataset, this simple architecture is going to fail on more complex tasks.
    The techniques that we will study in this chapter, will allow us to deal
    with those complex image recognition tasks, but also give us the necessary
    tools to deal with object detection and object segmentation.
  </p>
  <div class="separator" />
</Container>
