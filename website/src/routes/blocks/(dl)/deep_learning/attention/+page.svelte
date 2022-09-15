<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
</script>

<p class="danger">This chapter is in active development. Expect many changes in the next couple of weeks.</p>

<svelte:head>
  <title>World4AI | Deep Learning | Attention</title>
  <meta
    name="description"
    content="Since 2017 attention mechanism and transformers especially have taken the world by storm. Nowadays most modern deep learning sysmems use attention. In this chapter we study those systems."
  />
</svelte:head>

<h1>Attention</h1>
<div class="separator" />

<Container>
  <p>When we look at pictures or read text we do not focus on all the available data, but on some specific parts. In other words we pay attention to certain elements, while we disregard others. When you look at the circle below for example, your eye starts to focus on the red dot in the middle.</p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 500 500">
      <circle cx=250 cy=250 r=240 fill="var(--main-color-4)" stroke="black"/>
      <circle cx=250 cy=250 r=6 fill="var(--main-color-1)" stroke="black"/>
    </svg>
  </SvgContainer>
  <p>The attention mechanisms that we are going to study in this chapter are designed to learn what parts of the data the neural network has to pay attention to. Those neural networks have become state of the art in natural language processing and even computer vision (often surpassing convolutional neural networks). These models were originally developed for text (translation specifically), so that is where we are going to start our journey.</p>
  <div class="separator" />
</Container>

