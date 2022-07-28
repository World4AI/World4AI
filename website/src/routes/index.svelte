<svelte:head>
  <title>World4AI | A Place To Learn AI</title>
  <meta
    name="description"
    content="World4AI is a learning resource for all AI related topics. This includes machine learning, reinforcement learning, ai related mathematics, programming and much more."
  />
</svelte:head>

<div class="hero-message">
  <div class="border-container">
    <h1>World4AI</h1>
  </div>
  <h2>Learn AI block by block</h2>
</div>

<style>
  .hero-message {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .border-container {
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
    font-weight: 400;
    position: relative;
    letter-spacing: 7px;
    margin-bottom: 0;
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
    font-weight: 200;
    font-size: 40px;
    letter-spacing: 3px;
    text-transform: uppercase;
  }

  @media (max-width: 768px) {
    h1 {
      font-size: 30px;
    }

    h2 {
      font-size: 15px;
    }
  }
</style>
