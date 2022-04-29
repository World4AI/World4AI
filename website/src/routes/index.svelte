<script>
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | A Place To Learn And Advance in AI</title>
  <meta
    name="description"
    content="World4AI is a learning resource for all AI related topics. This includes machine learning, reinforcement learning, ai related mathematics, programming and much more."
  />
</svelte:head>

<Container>
  <div class="hero-message">
    <div class="border-container">
      <h1>World4AI</h1>
    </div>
    <h2>A place to learn AI</h2>
  </div>
</Container>
<div class="drawing-container" />

<style>
  .hero-message {
    margin-top: 170px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .border-container {
    display: flex;
    position: relative;
  }

  .border-container::before,
  .border-container::after {
    content: "";
    position: absolute;
    width: 60px;
    height: 3px;
    background-color: var(--text-color);
    top: 50%;
  }
  .border-container::before {
    left: -90px;
  }

  .border-container::after {
    right: -90px;
  }

  h1 {
    font-size: 90px;
    font-weight: normal;
    position: relative;
    letter-spacing: 7px;
  }

  h1::before,
  h1::after {
    content: "";
    position: absolute;
    width: 100%;
    height: 1px;
    background-color: var(--text-color);
    left: 0;
  }

  h1::before {
    top: 0;
  }

  h1::after {
    bottom: 0;
  }

  h2 {
    margin-top: 10px;
    font-style: italic;
    font-weight: normal;
    font-size: 45px;
    letter-spacing: 3px;
  }

  .drawing-container {
    margin-top: 50px;
    height: 500px;
    width: 100%;
    background: var(--main-color-4);
  }
</style>
