<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Sentence from "$lib/BlockLetters/Sentence.svelte";
  import Alert from "$lib/Alert.svelte";

  import { onMount } from "svelte";
  onMount(() => {
    let interval = setInterval(() => {
      offset -= 10;
    }, 300);

    return () => clearInterval(interval);
  });
  let offset = 0;
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

    <h2>Mathematics</h2>
    <Alert type="danger">Block not yet started</Alert>
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
      Computer vision, natural language processing, generative art, board games
      and many more areas have been taken over by deep learning. Once it became
      clear that we can scale neural networks to improve the performance in a
      particular area, other methods became obsolete. Within a decade from the
      so called ImageNet moment in 2012, deep learning became ubiquitous.
      Nowadays our mobile phones make sure that the pictures you take with your
      mobile phone are as crisp as those taken with a professional camera,
      recommendation systems like those used by Netflix or Spotify do a great
      job at finding new products based on your consumption history and Alexa by
      Amazon can understand your actual speach and trigger actions based on your
      words. Yet all those achievements will pale in comparison to those that
      will surely come in the next decades. Often it feels like we just barely
      scratched the surface of what is actually possible with those systems.
      This is an exciting time to study deep learning and this website will
      hopefully become a stepping stone in your journey towards deep learning
      mastery.
    </p>
    <p>
      In this block we are going to take a deep dive into deep learning. We will
      study the very first artificial neurons that were developed in the 1950's,
      the progress towards multilayer neural networks with the help of the
      backpropagation algorithm as well as state of the art deep learning.
    </p>
    <p>
      The block is suited for absolute beginners and we do not assume any prior
      knowledge of deep learning, but exposure to linear algebra, probability
      theory and calculus is assumed. We also expect fundamental Python
      knowledge, as theoretical lectures will be accompanied by practical Python
      books and exercises.
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

    <h2>Reinforcement Learning</h2>
    <Alert type="warning">
      The block has been started, but this is very early work in progress.
    </Alert>
    <p>
      A reasonable way to think about reinforcement learning is to think about
      animals whom you would like to teach good behaviour and discourage from
      bad behaviour. The way you would go about it would probably involve some
      reward when the animal behaves the way you desire and punishment when it
      doesn’t.
    </p>
    <SvgContainer maxWidth="700px">
      <svg
        version="1.1"
        viewBox="0 0 400 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        <g fill="none" class="stroke-black" stroke-linecap="square">
          <g id="lines" stroke-width="2px">
            <path d="m0 10v190h390" />
            <path d="m400 190v-190h-390" />
            <g>
              <path d="m15 70v-55" />
              <path d="m15 15h65" />
              <path d="m90 0v25" />
              <path d="m90 25h75" />
              <path d="m15 85v60" />
              <path d="m60 200v-25" />
              <path d="m15 175h75" />
              <path d="m35 35h75" />
              <path d="m70 35v25" />
              <path d="m90 35v25" />
              <path d="m50 55v25" />
              <path d="m35 80h90" />
              <path d="m185 0v80" />
              <path d="m185 40h-65" />
              <path d="m140 40v40" />
              <path d="m140 80h25" />
              <path d="m165 80v-30" />
              <path d="m35 95h215" />
              <path d="m200 95v-80" />
              <path d="m15 160h115" />
              <path d="m130 110v75" />
              <path d="m105 160v40" />
              <path d="m100 95v55" />
              <path d="m80 160v-55" />
              <path d="m60 95v55" />
              <path d="m40 160v-55" />
              <path d="m145 95v70" />
              <path d="m140 185h100" />
              <path d="m160 110h95v50h10v40" />
              <path d="m205 185v-75" />
              <path d="m225 110v65" />
              <path d="m185 110v65" />
              <path d="m165 185v-65" />
              <path d="m220 95v-60" />
              <path d="m215 15h70v85h85" />
              <path d="m250 15v-15" />
              <path d="m240 15v60" />
              <path d="m255 70v-35h30" />
              <path d="m300 85v-85" />
              <path d="m315 15v85" />
              <path d="m335 0v60" />
              <path d="m355 15v55" />
              <path d="m335 70h50" />
              <path d="m375 0v60" />
              <path d="m335 70v20" />
              <path d="m400 95h-20" />
              <path d="m355 100v-10" />
              <path d="m400 185h-35v-35h20" />
              <path d="m385 115v25h-60v-10h-40v-10" />
              <path d="m360 140v-20" />
              <path d="m340 115v25" />
              <path d="m265 80v70h60" />
              <path d="m280 150v35" />
              <path d="m295 200v-40h45v-10" />
              <path d="m340 170h-35v30" />
              <path d="m350 160v40" />
            </g>
          </g>
          <path
            id="path-bad"
            class="stroke-red-600"
            d="m5 10v160h90v10h-30v15h35v-30h-90v-15h15v-50h25v55h20v-55h20v55h25v-55h20v90h-10v-25h-15v30h30v-25h15v-70h105v55h15v40h10v-40h45v-10h25v50h40"
            stroke-dasharray="4, 4"
            stroke-dashoffset={offset}
          />
          <path
            id="path-good"
            class="stroke-blue-600"
            d="m10 10h75v20h30v40h15v15h65v-75h15v15h20v60h30v-30h15v55h120v35h-35v45h35"
            stroke-dasharray="4, 4"
            stroke-dashoffset={offset}
          />
        </g>
      </svg>
    </SvgContainer>
    <p>
      Let's consider mice in a maze that get food once they arrive at the goal.
      The amount of food could depend on the time a mouse needs to complete the
      maze. If the mouse takes the blue path, the route is relatively straight
      forward and the mouse ends up at the goal without any delays. The red path
      on the other hand leads to a dead end, which forces the mouse to retrace
      the steps. Over time the mouse will learn the shourtest route which
      maximizes the amount of food.
    </p>
    <p>
      In reinforcement learning we similarly deal with agents which have to
      learn certain behaviour. The world (environment) that the agent interacts
      with provides reinforcement in form of numeric rewards which can be used
      to learn the desired behaviour.
    </p>
    <div class="action">
      <a href="/blocks/reinforcement_learning/intuition/introduction">
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

    <h2>GOFAI</h2>
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
