<script>
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | A Place To Learn AI</title>
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
<div class="drawing-container">
  <div class="cards-container">
    <div class="card">
      <h3>Symbolic AI</h3>
      <svg
        version="1.1"
        viewBox="0 0 350 400"
        xmlns="http://www.w3.org/2000/svg"
      >
        <g id="symbols" fill="#fff" stroke="#000">
          <rect
            fill="var(--main-color-1)"
            id="square"
            x="76.535"
            y="179.46"
            width="194.98"
            height="48.745"
            stroke-width="1.026"
          />
          <circle
            fill="var(--main-color-3)"
            id="circle"
            cx="286.08"
            cy="337.3"
            r="60.189"
            stroke-width="1.026"
          />
          <path
            fill="var(--main-color-2)"
            id="triangle"
            transform="matrix(1.9183 0 0 2.0826 -264.96 -65.816)"
            d="m201.22 88.028h-62.45l31.225-54.083z"
            stroke-width=".51333"
          />
        </g>
      </svg>
    </div>
    <div class="card">
      <h3>Machine Learning</h3>
      <svg
        version="1.1"
        viewBox="0 0 350 400"
        xmlns="http://www.w3.org/2000/svg"
      >
        <g fill="none" stroke="#000" stroke-width="1.0175">
          <g id="left">
            <rect
              id="left-box"
              x="7.051"
              y="6.7799"
              width="60.322"
              height="386.44"
              ry="0"
            />
            <g id="left-circles" fill="var(--main-color-3)">
              <circle cx="36.958" cy="46.862" r="20.351" />
              <circle cx="37.975" cy="353.14" r="20.351" />
              <circle cx="37.212" cy="251.05" r="20.351" />
              <circle cx="37.212" cy="148.95" r="20.351" />
            </g>
          </g>
          <g id="middle">
            <rect
              id="middle-box"
              x="144.59"
              y="52.999"
              width="60.479"
              height="294"
              ry="0"
            />
            <g id="middle-circles" fill="var(--main-color-1)">
              <circle cx="174.83" cy="97.738" r="20.351" />
              <circle cx="174.83" cy="301.24" r="20.351" />
              <circle cx="174.83" cy="199.49" r="20.351" />
            </g>
          </g>
          <g id="right">
            <rect
              id="right-box"
              x="281.26"
              y="138.79"
              width="60.86"
              height="122.42"
              ry="0"
            />
            <g id="right-circles" fill="var(--main-color-2)">
              <circle cx="311.69" cy="168.97" r="20.351" />
              <circle cx="311.69" cy="229" r="20.351" />
            </g>
          </g>
          <g id="path" stroke-dasharray="2.03506, 4.0701">
            <g id="left-path">
              <path d="m67.992 98.12h76.315" />
              <path d="m67.992 301.88h76.315" />
              <path d="m67.992 200.13h76.315" />
            </g>
            <g id="right-path">
              <path d="m205.36 169.35h76.315" />
              <path d="m205.36 230.65h76.315" />
            </g>
          </g>
        </g>
      </svg>
    </div>
  </div>
</div>

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
    width: 100%;
    background: var(--main-color-4);
  }
  .cards-container {
    display: flex;
    justify-content: space-around;
  }

  .card {
    margin: 20px 0;
    padding: 0 10px;
    border-left: 1px solid black;
    border-right: 1px solid black;
    flex-basis: 350px;
    flex-grow: 0.02;
  }

  .card h3 {
    text-align: center;
    font-size: 30px;
    font-weight: normal;
    margin: 30px 20px;
    border-top: 1px solid black;
    border-bottom: 1px solid black;
  }

  @media (max-width: 768px) {
    h1 {
      font-size: 30px;
    }

    h2 {
      font-size: 20px;
    }

    .card {
      width: 250px;
      margin-bottom: 10px;
    }

    .cards-container {
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }

    .border-container::before,
    .border-container::after {
      width: 10px;
    }
    .border-container::before {
      left: -25px;
    }

    .border-container::after {
      right: -25px;
    }
  }
</style>
