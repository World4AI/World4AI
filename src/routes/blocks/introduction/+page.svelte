<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Sentence from "$lib/BlockLetters/Sentence.svelte";
  import Alert from "$lib/Alert.svelte";
</script>

<svelte:head>
  <title>World4AI | Artificial Intelligence Blocks</title>
  <meta
    name="description"
    content="World4AI is a organized into different blocks. Each block represents a topic that is relevant to artificial intelligence. For example deep learning, symbolic ai or neurosciece."
  />
</svelte:head>

<article>
  <h1>Blocks</h1>
  <div class="separator" />
  <Container>
    <p>
      World4AI is organized into building blocks. Each block forms knowledge
      that is fundamental to modern artificial intelligence.
    </p>

    <SvgContainer maxWidth={"700px"}>
      <Sentence sentence={"blocks"} />
    </SvgContainer>

    <p>
      We intend to cover the basics of deep learning, symbolic ai, neurosciece
      and much more, but at the moment only a few sections have been started and
      none is considered finalized. More chapters will be published in the next
      weeks and months. But the overall process will most likely take many
      years.
    </p>
    <div class="separator" />

    <h2>Deep Learning</h2>
    <Alert type="info">
      Still work in progress, but many chapters, especially early chapters, can
      be studied.
    </Alert>
    <p>
      Most of the recent success in artificial intelligence is due to deep
      learning.
    </p>
    <SvgContainer maxWidth="300px">
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
            <g id="left-circles" fill="var(--main-color-1)">
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
            <g id="middle-circles" fill="var(--main-color-2)">
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
            <g id="right-circles" fill="var(--main-color-3)">
              <circle cx="311.69" cy="168.97" r="20.351" />
              <circle cx="311.69" cy="229" r="20.351" />
            </g>
          </g>
          <g id="path" stroke-dasharray="4, 4" stroke-width="1px">
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
    </SvgContainer>
    <p>
      This block is very practical and will teach you image recognition, text
      understanding and image generation. This is a good starting point, if you
      don't have any prior AI knowledge.
    </p>
    <div class="action">
      <a href="/blocks/deep_learning/fundamentals/introduction">
        <!-- feather icon -->
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="48"
          height="48"
          viewBox="0 0 24 24"
          fill="var(--main-color-3)"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="feather feather-arrow-right-circle"
        >
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 16 16 12 12 8" /><line
            x1="8"
            y1="12"
            x2="16"
            y2="12"
          />
        </svg>
      </a>
    </div>
    <div class="separator" />

    <h2>Symbolic and Neurosymbolic AI</h2>
    <Alert type="danger">Block not yet started</Alert>
    <div class="separator" />

    <h2>Neurosciece</h2>
    <Alert type="danger">Block not yet started</Alert>
    <div class="separator" />
  </Container>
</article>

<style>
  .action {
    padding: 20px 5px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
</style>
