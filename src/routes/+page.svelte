<svelte:head>
  <title>World4AI | Free and Open Source AI Education</title>
  <meta
    name="description"
    content="World4AI is a free and open source education plattform for all AI related topics. This includes machine learning, reinforcement learning, symbolic ai, neuroscience, mathematics, programming and much more."
  />
</svelte:head>

<main>
  <article>
    <div class="hero-message">
      <div>
        <h2>Free and Open Source</h2>
        <h1><span>AI</span> Education</h1>
      </div>
    </div>
    <div class="action light-blue">
      <h2>Start building your knowledge</h2>
      <a href="/blocks/introduction">
        <!-- feather icon -->
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="var(--main-color-3)" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-right-circle">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 16 16 12 12 8">
          </polyline><line x1="8" y1="12" x2="16" y2="12"></line>
        </svg>
        </a>
    </div>
  </article>
</main>

<style>
  .hero-message {
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: var(--main-color-1);
  }

  h1 {
    font-size: 90px;
    font-weight: 600;
    position: relative;
    letter-spacing: 7px;
    margin-bottom: 0;
    color: var(--background-color);
  }

  h1 span {
    border: 3px solid black; 
    display: inline-block;
    margin-bottom: 10px;
    padding: 10px;
  }

  h2 {
    font-style: italic;
    font-weight: 400;
    font-size: 40px;
    letter-spacing: 3px;
    color: var(--background-color);
  }

  .action {
      margin-top: 100px;
      padding: 20px 5px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
  }

  .action h2 {
    color: var(--text-color)
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
