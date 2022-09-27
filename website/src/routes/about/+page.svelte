<script>
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | About</title>
  <meta
    name="description"
    content="World4AI is a teaching and learning plattform to learn and advance in artificial intelligence."
  />
</svelte:head>

<main>
  <article>
    <h1>What is World4AI?</h1>
    <div class="separator" />
    <Container>
      <p>World4AI is my answer to the question “How can I push my knowledge in artificial intelligence to an intermediary level, when I start out as a complete beginner?”. World4AI was designed with the philosophy "learning by teaching" in mind, so the main purpose of this website is to have a personalized blog that I can utilize to learn about AI.</p>
      <p>The blog is going to be structured in form of blocks, where individual blocks cover a different area of artificial intelligence. Each block is similar to a book on a topic like deep learning or symbolic ai, but with a stronger focus on interactive examples.</p>
      <div class="flex-center">
        <img width="50px" src="./logo/logo.svg" alt="World4AI Logo" />
      </div>
      <p>The logo of World4AI represents this idea of different blocks that come together to form a knowledge base in AI. </p>
      <p class="warning">At the moment only a few sections have been written. The whole website is still a work in progress.</p>
      <div class="separator" />
    </Container>
  </article>
</main>

<style>
  .flex-center {
    display: flex;
    justify-content: center;
  }
</style>
