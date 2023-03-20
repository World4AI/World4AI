<script>
  import Container from "$lib/Container.svelte";
  import plantImg from "./photo_plant.jpg";
</script>

<svelte:head>
  <title>Attention - World4AI</title>
  <meta
    name="description"
    content="Since 2017 attention mechanism and transformers especially have taken the world by storm. These mechanisms allows the model to focus only on the relevant parts of the input and disregard the other. This simple idea is at the core of most modern deep learning systems."
  />
</svelte:head>

<h1>Attention</h1>
<div class="separator" />

<Container>
  <p>
    When we look at pictures or read text we do not focus on all the available
    data with equal strength, rather we focus on some specific parts. In other
    words we pay attention to certain elements, while we disregard others.
  </p>
  <p>
    Look at the image below for example. What part of the image do you notice
    first? Most likely the yellow blossoms of the plant.
  </p>
  <figure class="flex flex-col justify-center items-center max-w-sm mx-auto">
    <img
      src={plantImg}
      class="rounded-xl object-contain"
      alt="A plant seen through a round window."
    />
    <figcaption class="text-sm text-center">
      Photo by <a
        rel="noreferrer"
        href="https://unsplash.com/@woodyyan?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText"
        target="_blank">Woody Yan</a
      >
      on
      <a
        rel="noreferrer"
        target="_blank"
        href="https://unsplash.com/photos/ARQv6dJG4Fg?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText"
        >Unsplash</a
      >
    </figcaption>
  </figure>
  <p>
    Did you notice the large leafs outside the window? Did you notice that the
    white circle that surrounds the window has a small defect at the top right?
    Your mind automatically wandered to the plant, filtering out the unnecessary
    noise that is not relevant to your understanding of the image.
  </p>
  <p>
    The attention mechanisms that we are going to study in this chapter are
    designed to learn what parts of the data the neural network has to pay
    attention to. Attention models have become state of the art in natural
    language processing and even computer vision. These models were originally
    developed for language translation, therefore that is where we are going to
    start our journey.
  </p>
  <div class="separator" />
</Container>
