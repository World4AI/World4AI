<script>
  import Highlight from "$lib/Highlight.svelte";
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | About</title>
  <meta
    name="description"
    content="World4AI is used as a personalized plattform to learn and advance in artificial intelligence."
  />
</svelte:head>

<p class="warning">
  At the moment only a few sections have been written. The whole website is a
  work in progress.
</p>

<h1>What is World4AI?</h1>
<div class="separator" />
<div class="flex-center">
  <img width="50px" src="./logo/logo.svg" alt="World4AI Logo" />
</div>
<Container>
  <p>
    World4AI is my answer to the question “How can I push my knowledge in
    artificial intelligence to an intermediary level?”. The main purpose of
    World4AI is for me to have a personalized blog that I can utilize to learn
    about AI. The blog is going to be structured in form of blocks, where
    individual blocks cover a different area of artificial intelligence.
  </p>

  <p>
    The logo of World4AI represents this idea of different blocks that come
    together to form a knowledge base.
  </p>
  <div class="separator" />
</Container>

<style>
  .flex-center {
    display: flex;
    justify-content: center;
  }
</style>
