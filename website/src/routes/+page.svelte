<svelte:head>
  <title>World4AI | Free and Open Source AI Education</title>
  <meta
    name="description"
    content="World4AI is a free and open source education plattform for all AI related topics. This includes machine learning, reinforcement learning, ai related mathematics, programming and much more."
  />
</svelte:head>

<div class="hero-message">
  <div class="border-container">
    <h1>World4AI</h1>
  </div>
  <h2>Free and open source AI education</h2>
</div>

<style>
  .hero-message {
    display: flex;
    flex-direction: column;
    align-items: center;
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
