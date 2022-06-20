<script>
  import { draw } from "svelte/transition";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Introduction</title>
  <meta
    name="description"
    content="Deep learning is the main contributer of the recent success of machine learning. This block is dedicated to the study of deep learning."
  />
</svelte:head>

<h1>Deep Learning</h1>
<div class="separator" />

<div class="test">
  <SvgContainer maxWidth={"500px"}>
    <svg version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
      <g
        id="connections"
        fill="none"
        stroke="var(--text-color)"
        stroke-width="0.6px"
      >
        <g id="level1connections">
          <path in:draw={{ duration: 1000 }} d="m25 125 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 175h150" />
          <path in:draw={{ duration: 1000 }} d="m25 225 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 275 150-100" />
          <path in:draw={{ duration: 1000 }} d="m25 325 150-150" />
          <path in:draw={{ duration: 1000 }} d="m25 375 150-200" />
          <path in:draw={{ duration: 1000 }} d="m25 425 150-250" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-300" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 150" />
          <path in:draw={{ duration: 1000 }} d="m25 75 150 100" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 75h150" />
          <path in:draw={{ duration: 1000 }} d="m25 125 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 175 150-100" />
          <path in:draw={{ duration: 1000 }} d="m25 225 150-150" />
          <path in:draw={{ duration: 1000 }} d="m25 275 150-200" />
          <path in:draw={{ duration: 1000 }} d="m25 325 150-250" />
          <path in:draw={{ duration: 1000 }} d="m25 375 150-300" />
          <path in:draw={{ duration: 1000 }} d="m25 425 150-350" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-400" />
          <path in:draw={{ duration: 1000 }} d="m25 175 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 225h150" />
          <path in:draw={{ duration: 1000 }} d="m25 275 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 325 150-100" />
          <path in:draw={{ duration: 1000 }} d="m25 375 150-150" />
          <path in:draw={{ duration: 1000 }} d="m25 425 150-200" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-250" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 200" />
          <path in:draw={{ duration: 1000 }} d="m25 75 150 150" />
          <path in:draw={{ duration: 1000 }} d="m25 125 150 100" />
          <path in:draw={{ duration: 1000 }} d="m25 225 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 275h150" />
          <path in:draw={{ duration: 1000 }} d="m25 325 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 375 150-100" />
          <path in:draw={{ duration: 1000 }} d="m25 425 150-150" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-200" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 250" />
          <path in:draw={{ duration: 1000 }} d="m25 75 150 200" />
          <path in:draw={{ duration: 1000 }} d="m25 125 150 150" />
          <path in:draw={{ duration: 1000 }} d="m25 175 150 100" />
          <path in:draw={{ duration: 1000 }} d="m25 375 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 425h150" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 400" />
          <path in:draw={{ duration: 1000 }} d="m25 75 150 350" />
          <path in:draw={{ duration: 1000 }} d="m25 125 150 300" />
          <path in:draw={{ duration: 1000 }} d="m25 175 150 250" />
          <path in:draw={{ duration: 1000 }} d="m25 225 150 200" />
          <path in:draw={{ duration: 1000 }} d="m25 275 150 150" />
          <path in:draw={{ duration: 1000 }} d="m25 325 150 100" />
          <path in:draw={{ duration: 1000 }} d="m25 75 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 125h150" />
          <path in:draw={{ duration: 1000 }} d="m25 175 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 225 150-100" />
          <path in:draw={{ duration: 1000 }} d="m25 275 150-150" />
          <path in:draw={{ duration: 1000 }} d="m25 325 150-200" />
          <path in:draw={{ duration: 1000 }} d="m25 375 150-250" />
          <path in:draw={{ duration: 1000 }} d="m25 425 150-300" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-350" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 100" />
          <path in:draw={{ duration: 1000 }} d="m25 325 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 375h150" />
          <path in:draw={{ duration: 1000 }} d="m25 425 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-100" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 350" />
          <path in:draw={{ duration: 1000 }} d="m25 75 150 300" />
          <path in:draw={{ duration: 1000 }} d="m25 125 150 250" />
          <path in:draw={{ duration: 1000 }} d="m25 175 150 200" />
          <path in:draw={{ duration: 1000 }} d="m25 225 150 150" />
          <path in:draw={{ duration: 1000 }} d="m25 275 150 100" />
          <path in:draw={{ duration: 1000 }} d="m25 275 150 50" />
          <path in:draw={{ duration: 1000 }} d="m25 325h150" />
          <path in:draw={{ duration: 1000 }} d="m25 375 150-50" />
          <path in:draw={{ duration: 1000 }} d="m25 425 150-100" />
          <path in:draw={{ duration: 1000 }} d="m25 475 150-150" />
          <path in:draw={{ duration: 1000 }} d="m25 25 150 300" />
          <path in:draw={{ duration: 1000 }} d="m25 75 150 250" />
          <path in:draw={{ duration: 1000 }} d="m25 125 150 200" />
          <path in:draw={{ duration: 1000 }} d="m25 175 150 150" />
          <path in:draw={{ duration: 1000 }} d="m25 225 150 100" />
        </g>
        <g id="level2connections">
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75h150" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 125 150-50" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 175 150-100"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 225 150-150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 275 150-200"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 325 150-250"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 375 150-300"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 425 150-350"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 125 150-50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 125h150" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 175 150-50" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 225 150-100"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 275 150-150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 325 150-200"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 375 150-250"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 425 150-300"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75 150 50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 175 150-50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 225h150" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 275 150-50" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 325 150-100"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 375 150-150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 425 150-200"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75 150 150" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 125 150 100"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 175 150 50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 275 150-50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 175h150" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 225 150-50" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 275 150-100"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 325 150-150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 375 150-200"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 425 150-250"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75 150 100" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 125 150 50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 225 150-50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 325h150" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 375 150-50" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 425 150-100"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75 150 250" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 125 150 200"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 175 150 150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 225 150 100"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 275 150 50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 375 150-50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 275h150" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 325 150-50" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 375 150-100"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 425 150-150"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75 150 200" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 125 150 150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 175 150 100"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 225 150 50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 325 150-50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 425h150" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 125 150 300"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 175 150 250"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 225 150 200"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 275 150 150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 325 150 100"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 375 150 50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75 150 350" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 375h150" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 425 150-50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 75 150 300" />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 125 150 250"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 175 150 200"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 225 150 150"
          />
          <path
            in:draw={{ duration: 1000, delay: 1000 }}
            d="m175 275 150 100"
          />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 325 150 50" />
          <path in:draw={{ duration: 1000, delay: 1000 }} d="m175 425 150-50" />
        </g>
        <g id="level3connections">
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 75 150 160" />
          <path
            in:draw={{ duration: 1000, delay: 2000 }}
            d="m325 125 150 110"
          />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 175 150 60" />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 225 150 10" />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 275 150-40" />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 325 150-90" />
          <path
            in:draw={{ duration: 1000, delay: 2000 }}
            d="m325 375 150-140"
          />
          <path
            in:draw={{ duration: 1000, delay: 2000 }}
            d="m325 425 150-190"
          />
          <path
            in:draw={{ duration: 1000, delay: 2000 }}
            d="m325 125 150 160"
          />
          <path
            in:draw={{ duration: 1000, delay: 2000 }}
            d="m325 175 150 110"
          />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 225 150 60" />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 275 150 10" />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 325 150-40" />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 375 150-90" />
          <path
            in:draw={{ duration: 1000, delay: 2000 }}
            d="m325 425 150-140"
          />
          <path in:draw={{ duration: 1000, delay: 2000 }} d="m325 75 150 210" />
        </g>
      </g>
      <g
        id="nodes"
        fill="var(--main-color-1)"
        stroke="var(--text-color)"
        stroke-width="2"
      >
        <g id="level1nodes">
          <rect x="10" y="10" width="30" height="30" />
          <rect x="10" y="60" width="30" height="30" />
          <rect x="10" y="110" width="30" height="30" />
          <rect x="10" y="160" width="30" height="30" />
          <rect x="10" y="210" width="30" height="30" />
          <rect x="10" y="260" width="30" height="30" />
          <rect x="10" y="310" width="30" height="30" />
          <rect x="10" y="360" width="30" height="30" />
          <rect x="10" y="410" width="30" height="30" />
          <rect x="10" y="460" width="30" height="30" />
        </g>
        <g id="level2nodes">
          <rect x="160" y="60" width="30" height="30" />
          <rect x="160" y="110" width="30" height="30" />
          <rect x="160" y="160" width="30" height="30" />
          <rect x="160" y="210" width="30" height="30" />
          <rect x="160" y="260" width="30" height="30" />
          <rect x="160" y="310" width="30" height="30" />
          <rect x="160" y="360" width="30" height="30" />
          <rect x="160" y="410" width="30" height="30" />
        </g>
        <g id="level3nodes">
          <rect x="310" y="60" width="30" height="30" />
          <rect x="310" y="110" width="30" height="30" />
          <rect x="310" y="160" width="30" height="30" />
          <rect x="310" y="210" width="30" height="30" />
          <rect x="310" y="260" width="30" height="30" />
          <rect x="310" y="310" width="30" height="30" />
          <rect x="310" y="360" width="30" height="30" />
          <rect x="310" y="410" width="30" height="30" />
        </g>
        <g id="level4nodes">
          <rect x="460" y="220" width="30" height="30" />
          <rect x="460" y="270" width="30" height="30" />
        </g>
      </g>
    </svg>
  </SvgContainer>
</div>

<Container>
  <div class="separator" />
  <p>
    Computer vision, natural language processing, generative art, board games
    and many more areas have been taken over by deep learning. Once it became
    clear that we can scale neural networks to improve the performance in a
    particular area, other methods became obsolete. Within a decade from the so
    called ImageNet moment in 2012, deep learning became ubiquitous. Nowadays
    our mobile phones make sure that the pictures you take with your mobile
    phone are as crisp as those taken with a professional camera, recommendation
    systems like those used by Netflix or Spotify do a great job at finding new
    products based on your consumption history and Alexa by Amazon can
    understand your actual speach and trigger actions based on your words. Yet
    all those achievements will pale in comparison to those that will surely
    come in the next decades. Often it feels like we just barely scratched the
    surface of what is actually possible with those systems. This is an exciting
    time to study deep learning and this website will hopefully become a
    stepping stone in your journey towards deep learning mastery.
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
    theory and calculus is assumed. We also expect fundamental Python knowledge,
    as theoretical lectures will be accompanied by practical Python books and
    exercises.
  </p>
  <div class="separator" />
</Container>

<style>
  .test {
    background-color: var(--main-color-4);
  }
</style>
