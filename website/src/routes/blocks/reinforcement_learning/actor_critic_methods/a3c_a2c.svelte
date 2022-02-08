<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";

  let config = {
    parameters: {
      0: { layer: 0, type: "input", count: 4, annotation: "Input" },
      1: { layer: 1, type: "fc", count: 7, input: [0] },
      2: { layer: 2, type: "fc", count: 5, input: [1] },
      3: { layer: 2, type: "fc", count: 5, input: [1] },
      4: {
        layer: 3,
        type: "fc",
        count: 5,
        input: [2],
      },
      5: {
        layer: 3,
        type: "fc",
        count: 5,
        input: [3],
      },
      6: {
        layer: 4,
        type: "fc",
        count: 2,
        input: [4],
      },
      7: {
        layer: 4,
        type: "fc",
        count: 1,
        input: [5],
      },
    },
  };
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | A2C and A3C</title>
  <meta
    name="description"
    content="A3C and A2C are advantage actor-critic methods that utilize several agents simultaneously to collect trajectories. These different experiences help to correlate trajectories to improve the performace."
  />
</svelte:head>

<h1>A3C and A2C</h1>
<Question
  >What are the asynchronous and the syncronous advantage actor critics?</Question
>
<div class="separator" />

<h2>
  <Highlight>A</Highlight>syncronous <Highlight>A</Highlight>dvantage <Highlight
    >A</Highlight
  >ctor-<Highlight>C</Highlight>ritic (A3C)
</h2>
<h3>Asynchronity</h3>
<p>
  Policy gradient methods and by extension actor-critic algorithms are on-policy
  algorithms. To see why this is the case let us look at the calculation of
  gradient that is used in the gradient ascent step.
</p>
<Latex
  >{String.raw`
    \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\Big[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t)  [R_{t+1} + V(S_{t+1}) - V(S_t)]\Big] 
  `}</Latex
>
<p>
  In order to calculate the gradient, the expectation has to be estimated based
  on trajectories that are sampled with the help of the policy <Latex
    >{String.raw`\pi_{\theta}`}</Latex
  >. The expression <Latex>{String.raw`\tau\sim\pi_{\theta}`}</Latex> means that
  the trajectories are drawn from a distribution that depends on the policy <Latex
    >\pi</Latex
  >, which in turn depends on the neural network parameters <Latex>\theta</Latex
  >. Each time we perform a gradient ascent step the weights <Latex
    >\theta</Latex
  > are adjusted, which changes the policy and therefore changes the distribution.
  After each optimization step we generate a new policy and old experience tuples
  have to be thrown away, because those tuples were drawn from a different policy
  with different weights.
</p>

<p>
  Because of its off-policy nature we are not going to use the memory buffer
  that was used with value based reinforcement learning algorithms. But the
  memory buffer tried to solve a problem that is common in reinforcement
  learning: the highly correlated observations. The experience replay techniques
  draws random experiences from the memory buffer. This randomness increases the
  likelihood, that tuples from different episodes are drawn, which reduces the
  correlation between the observations.
</p>
<p>
  Policy gradient algorithms use tuples from the same trajectory <Latex
    >\tau</Latex
  >, when applying gradient ascent. Therefore we face highly corrrelated
  observations, which might destablelize training.
</p>
<p>
  The asynchronous advantage actor-critic (A3C) tries to solve the problem by
  running the same algorithm in parallel on different processor cores. Each core
  has a distinct agent that interacts with its own instance of the environment
  and updates the policy periodically. It is reasonable to assume that each
  agent faces different states and rewards when interacting with the
  environment, thus reducing the correlation problem.
</p>
<svg version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="Arrow1Mstart" overflow="visible" orient="auto">
      <path
        transform="scale(.4) translate(10)"
        d="m0 0 5-5-17.5 5 17.5 5-5-5z"
        fill="var(--text-color)"
        fill-rule="evenodd"
        stroke="var(--text-color)"
        stroke-width="1pt"
      />
    </marker>
    <marker id="marker27437" overflow="visible" orient="auto">
      <path
        transform="scale(.6) rotate(180) translate(0)"
        d="m8.7186 4.0337-10.926-4.0177 10.926-4.0177c-1.7455 2.3721-1.7354 5.6175-6e-7 8.0354z"
        fill="var(--text-color)"
        fill-rule="evenodd"
        stroke="var(--text-color)"
        stroke-width=".625"
      />
    </marker>
    <marker id="marker3806" overflow="visible" orient="auto">
      <path
        transform="scale(.8)"
        d="m0-7.0711-7.0711 7.0711 7.0711 7.0711 7.071-7.0711-7.071-7.0711z"
        fill="var(--text-color)"
        stroke="var(--text-color)"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g stroke="var(--text-color)">
    <g id="value-function" fill="none" stroke-linejoin="round">
      <rect
        x="49.716"
        y="243.13"
        width="46.333"
        height="46.368"
        opacity=".999"
      />
      <path
        d="m61.299 251.83 11.583 28.98 11.583-28.98"
        stroke-linecap="round"
        stroke-width="5"
      />
    </g>
    <g id="policy" fill="none">
      <rect
        x="404.83"
        y="243.13"
        width="46.333"
        height="46.368"
        opacity=".999"
        stroke-linejoin="round"
      />
      <path d="m416.42 281.53v-28.98h23.167v28.98" stroke-width="5" />
    </g>
    <g id="value-function-drawing" stroke-linecap="square" stroke-width="1px">
      <path d="m6.2647 353.29h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m22.919 353.29h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m39.574 353.29h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m122.85 303.17h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m6.2647 303.17h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m22.919 303.17h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m56.228 303.17h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m39.574 303.17h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m89.537 303.17h16.655v16.708h-16.655v-16.708" fill="#ff8080" />
      <path d="m72.883 303.17h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m106.19 303.17h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m39.574 319.88h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m6.2647 319.88h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m72.883 319.88h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m56.228 319.88h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m122.85 319.88h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m89.537 319.88h16.655v16.708h-16.655v-16.708" fill="#ff8080" />
      <path d="m22.919 319.88h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m106.19 319.88h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m106.19 336.58h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m56.228 336.58h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m22.919 336.58h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m6.2647 336.58h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m39.574 336.58h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m122.85 336.58h16.655v16.708h-16.655v-16.708" fill="#ff8080" />
      <path d="m89.537 336.58h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m72.883 336.58h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m56.228 353.29h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m72.883 353.29h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m89.537 353.29h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <g fill="#fd5">
        <path d="m106.19 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m122.85 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m139.5 386.71h-16.655v-16.708h16.655v16.708" />
      </g>
      <path d="m122.85 386.71h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m106.19 386.71h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m22.919 436.83h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m139.5 436.83h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m122.85 436.83h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m89.537 436.83h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m106.19 436.83h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m56.228 436.83h-16.655v-16.708h16.655v16.708" fill="#ff8080" />
      <path d="m72.883 436.83h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m39.574 436.83h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m106.19 420.12h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m139.5 420.12h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m72.883 420.12h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m89.537 420.12h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m22.919 420.12h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m56.228 420.12h-16.655v-16.708h16.655v16.708" fill="#ff8080" />
      <path d="m122.85 420.12h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m39.574 420.12h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m39.574 403.42h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m89.537 403.42h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m122.85 403.42h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m139.5 403.42h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m106.19 403.42h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m22.919 403.42h-16.655v-16.708h16.655v16.708" fill="#ff8080" />
      <path d="m56.228 403.42h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m72.883 403.42h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m89.537 386.71h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m72.883 386.71h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m56.228 386.71h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m39.574 386.71h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m22.919 386.71h-16.655v-16.708h16.655v16.708" fill="#fd5" />
    </g>
    <g id="policy-drawing" fill="none" stroke-width="1px">
      <g stroke-linecap="square">
        <path d="m361.38 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m361.38 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m428 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m361.38 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m428 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m361.38 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m428 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m428 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m494.62 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m494.62 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m428 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m494.62 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m428 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m494.62 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m428 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m428 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 386.71h-16.655v-16.708h16.655v16.708" />
      </g>
      <g>
        <path d="m372 311.94h16.655" marker-end="url(#marker27437)" />
        <path d="m369.58 328.93v16.655" marker-end="url(#marker27437)" />
        <path d="m371.85 345.43h16.655" marker-end="url(#marker27437)" />
        <path d="m403 378.94h16.655" marker-end="url(#marker27437)" />
        <path d="m419.5 361.44h16.655" marker-end="url(#marker27437)" />
        <path d="m467.5 429.44h16.655" marker-end="url(#marker27437)" />
        <path d="m451 328.19h16.655" marker-end="url(#marker27437)" />
        <path d="m418.98 312.69v16.654" marker-end="url(#marker27437)" />
        <path d="m485.98 311.42h-16.655" marker-end="url(#marker27437)" />
        <path d="m385.73 349.44v16.655" marker-end="url(#marker27437)" />
        <path d="m369.48 379.94v16.655" marker-end="url(#marker27437)" />
        <path d="m466.73 311.42h-16.655" marker-end="url(#marker27437)" />
        <path d="m438.98 328.42h-16.655" marker-end="url(#marker27437)" />
        <path d="m385.85 394.68h16.655" marker-end="url(#marker27437)" />
        <path d="m370.1 411.93h16.655" marker-end="url(#marker27437)" />
        <path d="m419.73 396.19v16.655" marker-end="url(#marker27437)" />
        <path d="m486.48 361.67h-16.655" marker-end="url(#marker27437)" />
        <path d="m452.48 361.19v16.655" marker-end="url(#marker27437)" />
        <path d="m468.98 394.69v16.655" marker-end="url(#marker27437)" />
        <path d="m438.23 415.44v16.655" marker-end="url(#marker27437)" />
        <path d="m437.75 395.19h16.655" marker-end="url(#marker27437)" />
        <path d="m403.75 429.19h16.655" marker-end="url(#marker27437)" />
      </g>
    </g>
    <g
      id="interactions"
      fill="none"
      stroke-dasharray="0.483196, 0.483196"
      stroke-width=".4832"
    >
      <path
        d="m120.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m49.737 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m230.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m159.74 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m340.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m269.74 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m450.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m379.74 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
    </g>
    <g id="value-copy" fill="none" stroke-dasharray="1, 1">
      <path
        d="m60 230c0-9.999 0-19.999 3.3339-25s10-5.0005 13.333-11.667c3.3328-6.6663 3.3328-19.999 3.3328-33.334"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
      <path
        d="m70 230c0-6.6657 0-13.332 13.334-16.666 13.334-3.3338 40-3.3338 53.333-13.334 13.333-9.9997 13.333-29.999 16.667-40s10-10 16.666-10"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
      <path
        d="m90 230c33.334 0 66.668 0 83.334-8.333 16.666-8.333 16.666-24.999 36.667-33.333 20.001-8.3338 60-8.3338 80-11.667s20-9.9994 20-16.667"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
      <path
        d="m110 260c63.334 0 126.67 0 158.33-11.666 31.666-11.666 31.666-34.999 51.667-46.666 20.001-11.667 60-11.667 80-16.667s20-14.999 20-25"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
    </g>
    <g id="policy-copy" fill="none" stroke-dasharray="1, 1">
      <path
        d="m400 270c-43.332 0-86.666 0-108.33-11.666-21.667-11.666-21.667-34.999-51.667-46.666-30-11.667-89.999-11.667-120-18.333s-30-19.999-30-33.334"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
      <path
        d="m400 260c-26.666 0-53.332 0-66.666-11.666-13.334-11.666-13.334-34.999-35-46.666-21.667-11.667-64.999-11.667-86.666-16.667-21.667-4.9996-21.667-14.999-21.667-25"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
      <path
        d="m420 240c0-13.332 0-26.666-16.094-33.215-16.094-6.5493-48.281-6.3136-64.612-12.711-16.331-6.3971-16.802-19.427-17.274-32.459"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
      <path
        d="m430 240c0-19.999 0-39.999 5.0006-50s15-10 20-16.667 4.9995-19.999 3.3333-26.667c-1.6662-6.6672-4.9995-6.6672-8.3338-6.6672"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
      />
    </g>
    <g id="agents" fill="none">
      <g>
        <rect
          x="70.769"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="73.959"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="86.718"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m76 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m82.667 103.33v-13.333" />
        <path d="m89.333 103.33v-13.333" />
        <path d="m96 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m102.67 110h6.6667v-6.6667h6.6667" />
        <path d="m102.67 116.67h13.333" />
        <path d="m102.67 123.33h13.333" />
        <path d="m102.67 130h6.6667v6.6667h6.6667" />
        <path d="m69.333 110h-6.6667v-6.6667h-6.6667" />
        <path d="m69.333 116.67h-13.333" />
        <path d="m69.333 123.33h-13.333" />
        <path d="m69.333 130h-6.6667v6.6667h-6.6667" />
        <path d="m76 136.67v6.6667h-6.6667v6.6667" />
        <path d="m82.667 136.67v13.333" />
        <path d="m89.333 136.67v13.333" />
        <path d="m96 136.67v6.6667h6.6667v6.6667" />
      </g>
      <g>
        <rect
          x="180.77"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="183.96"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="196.72"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m186 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m192.67 103.33v-13.333" />
        <path d="m199.33 103.33v-13.333" />
        <path d="m206 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m212.67 110h6.6667v-6.6667h6.6667" />
        <path d="m212.67 116.67h13.333" />
        <path d="m212.67 123.33h13.333" />
        <path d="m212.67 130h6.6667v6.6667h6.6667" />
        <path d="m179.33 110h-6.6667v-6.6667h-6.6667" />
        <path d="m179.33 116.67h-13.333" />
        <path d="m179.33 123.33h-13.333" />
        <path d="m179.33 130h-6.6667v6.6667h-6.6667" />
        <path d="m186 136.67v6.6667h-6.6667v6.6667" />
        <path d="m192.67 136.67v13.333" />
        <path d="m199.33 136.67v13.333" />
        <path d="m206 136.67v6.6667h6.6667v6.6667" />
      </g>
      <g>
        <rect
          x="288.77"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="291.96"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="304.72"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m294 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m300.67 103.33v-13.333" />
        <path d="m307.33 103.33v-13.333" />
        <path d="m314 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m320.67 110h6.6667v-6.6667h6.6667" />
        <path d="m320.67 116.67h13.333" />
        <path d="m320.67 123.33h13.333" />
        <path d="m320.67 130h6.6667v6.6667h6.6667" />
        <path d="m287.33 110h-6.6667v-6.6667h-6.6667" />
        <path d="m287.33 116.67h-13.333" />
        <path d="m287.33 123.33h-13.333" />
        <path d="m287.33 130h-6.6667v6.6667h-6.6667" />
        <path d="m294 136.67v6.6667h-6.6667v6.6667" />
        <path d="m300.67 136.67v13.333" />
        <path d="m307.33 136.67v13.333" />
        <path d="m314 136.67v6.6667h6.6667v6.6667" />
      </g>
      <g>
        <rect
          x="398.77"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="401.96"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="414.72"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m404 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m410.67 103.33v-13.333" />
        <path d="m417.33 103.33v-13.333" />
        <path d="m424 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m430.67 110h6.6667v-6.6667h6.6667" />
        <path d="m430.67 116.67h13.333" />
        <path d="m430.67 123.33h13.333" />
        <path d="m430.67 130h6.6667v6.6667h6.6667" />
        <path d="m397.33 110h-6.6667v-6.6667h-6.6667" />
        <path d="m397.33 116.67h-13.333" />
        <path d="m397.33 123.33h-13.333" />
        <path d="m397.33 130h-6.6667v6.6667h-6.6667" />
        <path d="m404 136.67v6.6667h-6.6667v6.6667" />
        <path d="m410.67 136.67v13.333" />
        <path d="m417.33 136.67v13.333" />
        <path d="m424 136.67v6.6667h6.6667v6.6667" />
      </g>
    </g>
    <g id="envs" fill="none">
      <rect
        x="55.297"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="60.248"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m65.198 11.248v49.505" />
        <path d="m70.149 11.248v49.505" />
        <path d="m75.099 11.248v49.505" />
        <path d="m80.05 11.248v49.505" />
        <path d="m85 11.248v49.505" />
        <path d="m89.95 11.248v49.505" />
        <path d="m94.901 11.248v49.505" />
        <path d="m99.851 11.248v49.505" />
        <path d="m104.8 11.248v49.505" />
        <path d="m60.248 16.198h49.505" />
        <path d="m60.248 21.149h49.505" />
        <path d="m60.248 26.099h49.505" />
        <path d="m60.248 31.05h49.505" />
        <path d="m60.248 36h49.505" />
        <path d="m60.248 40.95h49.505" />
        <path d="m60.248 45.901h49.505" />
        <path d="m60.248 50.851h49.505" />
        <path d="m60.248 55.802h49.505" />
      </g>
      <rect
        x="165.3"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="170.25"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m175.2 11.248v49.505" />
        <path d="m180.15 11.248v49.505" />
        <path d="m185.1 11.248v49.505" />
        <path d="m190.05 11.248v49.505" />
        <path d="m195 11.248v49.505" />
        <path d="m199.95 11.248v49.505" />
        <path d="m204.9 11.248v49.505" />
        <path d="m209.85 11.248v49.505" />
        <path d="m214.8 11.248v49.505" />
        <path d="m170.25 16.198h49.505" />
        <path d="m170.25 21.149h49.505" />
        <path d="m170.25 26.099h49.505" />
        <path d="m170.25 31.05h49.505" />
        <path d="m170.25 36h49.505" />
        <path d="m170.25 40.95h49.505" />
        <path d="m170.25 45.901h49.505" />
        <path d="m170.25 50.851h49.505" />
        <path d="m170.25 55.802h49.505" />
      </g>
      <rect
        x="275.3"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="280.25"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m285.2 11.248v49.505" />
        <path d="m290.15 11.248v49.505" />
        <path d="m295.1 11.248v49.505" />
        <path d="m300.05 11.248v49.505" />
        <path d="m305 11.248v49.505" />
        <path d="m309.95 11.248v49.505" />
        <path d="m314.9 11.248v49.505" />
        <path d="m319.85 11.248v49.505" />
        <path d="m324.8 11.248v49.505" />
        <path d="m280.25 16.198h49.505" />
        <path d="m280.25 21.149h49.505" />
        <path d="m280.25 26.099h49.505" />
        <path d="m280.25 31.05h49.505" />
        <path d="m280.25 36h49.505" />
        <path d="m280.25 40.95h49.505" />
        <path d="m280.25 45.901h49.505" />
        <path d="m280.25 50.851h49.505" />
        <path d="m280.25 55.802h49.505" />
      </g>
      <rect
        x="385.3"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="390.25"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m395.2 11.248v49.505" />
        <path d="m400.15 11.248v49.505" />
        <path d="m405.1 11.248v49.505" />
        <path d="m410.05 11.248v49.505" />
        <path d="m415 11.248v49.505" />
        <path d="m419.95 11.248v49.505" />
        <path d="m424.9 11.248v49.505" />
        <path d="m429.85 11.248v49.505" />
        <path d="m434.8 11.248v49.505" />
        <path d="m390.25 16.198h49.505" />
        <path d="m390.25 21.149h49.505" />
        <path d="m390.25 26.099h49.505" />
        <path d="m390.25 31.05h49.505" />
        <path d="m390.25 36h49.505" />
        <path d="m390.25 40.95h49.505" />
        <path d="m390.25 45.901h49.505" />
        <path d="m390.25 50.851h49.505" />
        <path d="m390.25 55.802h49.505" />
      </g>
    </g>
  </g>
</svg>
<p>
  At the core of the A3C algorithm as an asynchronous learning mechanism. The
  value and the policy function (implemented through a neural network) is
  initialized with shared weights that can be accessed by other processes on
  presumably different cores. Each agent that is spawned on a different process
  copies the shared weights into the local functions before each interaction
  sequence. After a certain number of steps or when the agent encounters the
  terminal state, the agent runs the gradient descent step on the global shared
  weights and copies the global shared weights into the local policy and value
  functions. The copying and updating is done in an asynchronous, non blocking,
  way. That means that there is no communication and coordination of individual
  agents. Sometimes the gradients that are copied back into the global shared
  value and policy network might be overridden by a different agent. This
  algorithm, called Hogwild, still manages to work well, even when the agents
  that live on different processes do not communicate with each other.
</p>
<p>
  The A3C algorithm uses different cores of the CPU to spawn different agents
  and environments. The experience batches are small and there is no need to use
  any GPUs, therefore the forward and backward passes are conducted exclusively
  on the CPU.
</p>
<h3>Advantage</h3>
<p>
  The A3C algorithm uses n-steps to calculate the advantage that is used in the
  calculation of the gradient. If the environment terminates before the n'th
  step the optimization is done with the data that is available. In the original
  paper the authors use 5 steps. The rest of the trajectory return is calculated
  through bootstrapping, which makes the algorithm an actor-critic algorithm.
  Below we can see the calculation of the advantage for each of the 5 steps.
</p>
<Latex
  >{String.raw`
   \begin{aligned}
   & A_{t+0} = R_{t+1} + R_{t+2} + R_{t+3} + R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t}) \\
   & A_{t+1} = R_{t+2} + R_{t+3} + R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t+1}) \\
   & A_{t+2} = R_{t+3} + R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t+2}) \\
   & A_{t+3} = R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t+3}) \\
   & A_{t+4} = R_{t+5} + V(S_{t+5}) - V(S_{t+4}) \\
   \end{aligned}
  `}</Latex
>
<p>
  The reason why the expression <Latex
    >{String.raw`R_t + V(S_{t+1}) - V(S_{t})`}</Latex
  > is called the advantage is because <Latex
    >{String.raw`R_t + V(S_{t+1})`}</Latex
  > can be seen as an estimate for <Latex>{String.raw`Q(s, a)`}</Latex> and<Latex
    >{String.raw`A(s, a) = Q(s, a) - V(s)`}</Latex
  >.
</p>

<h3>Entropy</h3>
<p>In A3C we use an additional entropy term to calculate the gradient.</p>
<Latex
  >{String.raw`
    \sum_t^H \Big[\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)A_t + \beta\nabla H(\pi_{\theta}(s_t))]\Big] 
  `}</Latex
>
<p>
  The entropy <Latex>H</Latex> measures how much uncertainty is inherent in a probability
  distribution. The more certain the actions become the lower the entropy gets. If
  we assume for example a policy with two actions, then a policy which tends to select
  the same action for a given state with almost 100% probability will have a very
  low entropy. If on the other hand both actions will be selected with 50% probability
  for a given state, then the entropy will be high.
</p>
<p>
  <Latex>H</Latex> calculates the entropy and beta <Latex>\beta</Latex> is used to
  measure the importance of the entropy. The reason for using entropy is to encourage
  exploration. The general idea of the policy gradient algorithm is to maximize actions
  with the highest advantage, but if the convergence to certain actions happen too
  soon, it is possible that the agent misses on more favorable actions. Higher entropy
  forces the policy function to contain more uncertainty and therefore explore more.
</p>

<h3>Combined Architecture</h3>
<NeuralNetwork {config} />
<p>
  So far we have used separate networks for the agent and the critic. A3C uses
  the same neural network for the initial layers and only the last layers are
  separated into the policy and the value outputs. This approach is especially
  useful when several convolutional layers have to be trained in order to
  evaluate images and the weight sharing might facilitate training.
</p>
<div class="separator" />
<h2>
  <Highlight>A</Highlight>dvantage <Highlight>A</Highlight>ctor-<Highlight
    >C</Highlight
  >ritic (A2C)
</h2>
<h3>Synchronity</h3>
<svg version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="Arrow1Mstart" overflow="visible" orient="auto">
      <path
        transform="scale(.4) translate(10)"
        d="m0 0 5-5-17.5 5 17.5 5-5-5z"
        fill="context-stroke"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
    <marker id="marker27437" overflow="visible" orient="auto">
      <path
        transform="scale(.6) rotate(180) translate(0)"
        d="m8.7186 4.0337-10.926-4.0177 10.926-4.0177c-1.7455 2.3721-1.7354 5.6175-6e-7 8.0354z"
        fill="context-stroke"
        fill-rule="evenodd"
        stroke-linejoin="round"
        stroke-width=".625"
      />
    </marker>
    <marker id="marker3806" overflow="visible" orient="auto">
      <path
        transform="scale(.8)"
        d="m0-7.0711-7.0711 7.0711 7.0711 7.0711 7.071-7.0711-7.071-7.0711z"
        fill="#fff"
        fill-rule="evenodd"
        stroke="#000"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g stroke="var(--text-color)">
    <g id="value-function" fill="none" stroke-linejoin="round">
      <rect
        x="49.716"
        y="243.13"
        width="46.333"
        height="46.368"
        opacity=".999"
      />
      <path
        d="m61.299 251.83 11.583 28.98 11.583-28.98"
        stroke-linecap="round"
        stroke-width="5"
      />
    </g>
    <g id="policy" fill="none">
      <rect
        x="404.83"
        y="243.13"
        width="46.333"
        height="46.368"
        opacity=".999"
        stroke-linejoin="round"
      />
      <path d="m416.42 281.53v-28.98h23.167v28.98" stroke-width="5" />
    </g>
    <g id="value-function-drawing" stroke-linecap="square" stroke-width="1px">
      <path d="m6.2647 353.29h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m22.919 353.29h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m39.574 353.29h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m122.85 303.17h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m6.2647 303.17h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m22.919 303.17h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m56.228 303.17h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m39.574 303.17h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m89.537 303.17h16.655v16.708h-16.655v-16.708" fill="#ff8080" />
      <path d="m72.883 303.17h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m106.19 303.17h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m39.574 319.88h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m6.2647 319.88h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m72.883 319.88h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m56.228 319.88h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m122.85 319.88h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m89.537 319.88h16.655v16.708h-16.655v-16.708" fill="#ff8080" />
      <path d="m22.919 319.88h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m106.19 319.88h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m106.19 336.58h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m56.228 336.58h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m22.919 336.58h16.655v16.708h-16.655v-16.708" fill="#fd5" />
      <path d="m6.2647 336.58h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m39.574 336.58h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <path d="m122.85 336.58h16.655v16.708h-16.655v-16.708" fill="#ff8080" />
      <path d="m89.537 336.58h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m72.883 336.58h16.655v16.708h-16.655v-16.708" fill="#59f" />
      <path d="m56.228 353.29h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m72.883 353.29h16.655v16.708h-16.655v-16.708" fill="#f59" />
      <path d="m89.537 353.29h16.655v16.708h-16.655v-16.708" fill="#acf" />
      <g fill="#fd5">
        <path d="m106.19 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m122.85 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m139.5 386.71h-16.655v-16.708h16.655v16.708" />
      </g>
      <path d="m122.85 386.71h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m106.19 386.71h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m22.919 436.83h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m139.5 436.83h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m122.85 436.83h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m89.537 436.83h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m106.19 436.83h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m56.228 436.83h-16.655v-16.708h16.655v16.708" fill="#ff8080" />
      <path d="m72.883 436.83h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m39.574 436.83h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m106.19 420.12h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m139.5 420.12h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m72.883 420.12h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m89.537 420.12h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m22.919 420.12h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m56.228 420.12h-16.655v-16.708h16.655v16.708" fill="#ff8080" />
      <path d="m122.85 420.12h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m39.574 420.12h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m39.574 403.42h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m89.537 403.42h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m122.85 403.42h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m139.5 403.42h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m106.19 403.42h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m22.919 403.42h-16.655v-16.708h16.655v16.708" fill="#ff8080" />
      <path d="m56.228 403.42h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m72.883 403.42h-16.655v-16.708h16.655v16.708" fill="#59f" />
      <path d="m89.537 386.71h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m72.883 386.71h-16.655v-16.708h16.655v16.708" fill="#f59" />
      <path d="m56.228 386.71h-16.655v-16.708h16.655v16.708" fill="#acf" />
      <path d="m39.574 386.71h-16.655v-16.708h16.655v16.708" fill="#fd5" />
      <path d="m22.919 386.71h-16.655v-16.708h16.655v16.708" fill="#fd5" />
    </g>
    <g id="policy-drawing" fill="none" stroke-width="1px">
      <g stroke-linecap="square">
        <path d="m361.38 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m361.38 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m428 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 303.17h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m361.38 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m428 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 319.88h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m378.04 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m361.38 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m394.69 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m428 336.58h16.655v16.708h-16.655v-16.708" />
        <path d="m411.35 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m428 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m444.65 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m461.31 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m477.96 353.29h16.655v16.708h-16.655v-16.708" />
        <path d="m494.62 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m494.62 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m428 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 436.83h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m494.62 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m428 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 420.12h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m477.96 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m494.62 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m461.31 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m428 403.42h-16.655v-16.708h16.655v16.708" />
        <path d="m444.65 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m428 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m411.35 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m394.69 386.71h-16.655v-16.708h16.655v16.708" />
        <path d="m378.04 386.71h-16.655v-16.708h16.655v16.708" />
      </g>
      <g>
        <path d="m372 311.94h16.655" marker-end="url(#marker27437)" />
        <path d="m369.58 328.93v16.655" marker-end="url(#marker27437)" />
        <path d="m371.85 345.43h16.655" marker-end="url(#marker27437)" />
        <path d="m403 378.94h16.655" marker-end="url(#marker27437)" />
        <path d="m419.5 361.44h16.655" marker-end="url(#marker27437)" />
        <path d="m467.5 429.44h16.655" marker-end="url(#marker27437)" />
        <path d="m451 328.19h16.655" marker-end="url(#marker27437)" />
        <path d="m418.98 312.69v16.654" marker-end="url(#marker27437)" />
        <path d="m485.98 311.42h-16.655" marker-end="url(#marker27437)" />
        <path d="m385.73 349.44v16.655" marker-end="url(#marker27437)" />
        <path d="m369.48 379.94v16.655" marker-end="url(#marker27437)" />
        <path d="m466.73 311.42h-16.655" marker-end="url(#marker27437)" />
        <path d="m438.98 328.42h-16.655" marker-end="url(#marker27437)" />
        <path d="m385.85 394.68h16.655" marker-end="url(#marker27437)" />
        <path d="m370.1 411.93h16.655" marker-end="url(#marker27437)" />
        <path d="m419.73 396.19v16.655" marker-end="url(#marker27437)" />
        <path d="m486.48 361.67h-16.655" marker-end="url(#marker27437)" />
        <path d="m452.48 361.19v16.655" marker-end="url(#marker27437)" />
        <path d="m468.98 394.69v16.655" marker-end="url(#marker27437)" />
        <path d="m438.23 415.44v16.655" marker-end="url(#marker27437)" />
        <path d="m437.75 395.19h16.655" marker-end="url(#marker27437)" />
        <path d="m403.75 429.19h16.655" marker-end="url(#marker27437)" />
      </g>
    </g>
    <g
      id="interactions"
      fill="none"
      stroke-dasharray="0.483196, 0.483196"
      stroke-width=".4832"
    >
      <path
        d="m120.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m49.737 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m230.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m159.74 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m340.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m269.74 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m450.26 36.095h14.496v82.143h-14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        d="m379.74 118.24h-14.496v-82.143h14.496"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
    </g>
    <g id="agents" fill="none">
      <g>
        <rect
          x="70.769"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="73.959"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="86.718"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m76 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m82.667 103.33v-13.333" />
        <path d="m89.333 103.33v-13.333" />
        <path d="m96 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m102.67 110h6.6667v-6.6667h6.6667" />
        <path d="m102.67 116.67h13.333" />
        <path d="m102.67 123.33h13.333" />
        <path d="m102.67 130h6.6667v6.6667h6.6667" />
        <path d="m69.333 110h-6.6667v-6.6667h-6.6667" />
        <path d="m69.333 116.67h-13.333" />
        <path d="m69.333 123.33h-13.333" />
        <path d="m69.333 130h-6.6667v6.6667h-6.6667" />
        <path d="m76 136.67v6.6667h-6.6667v6.6667" />
        <path d="m82.667 136.67v13.333" />
        <path d="m89.333 136.67v13.333" />
        <path d="m96 136.67v6.6667h6.6667v6.6667" />
      </g>
      <g>
        <rect
          x="180.77"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="183.96"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="196.72"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m186 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m192.67 103.33v-13.333" />
        <path d="m199.33 103.33v-13.333" />
        <path d="m206 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m212.67 110h6.6667v-6.6667h6.6667" />
        <path d="m212.67 116.67h13.333" />
        <path d="m212.67 123.33h13.333" />
        <path d="m212.67 130h6.6667v6.6667h6.6667" />
        <path d="m179.33 110h-6.6667v-6.6667h-6.6667" />
        <path d="m179.33 116.67h-13.333" />
        <path d="m179.33 123.33h-13.333" />
        <path d="m179.33 130h-6.6667v6.6667h-6.6667" />
        <path d="m186 136.67v6.6667h-6.6667v6.6667" />
        <path d="m192.67 136.67v13.333" />
        <path d="m199.33 136.67v13.333" />
        <path d="m206 136.67v6.6667h6.6667v6.6667" />
      </g>
      <g>
        <rect
          x="288.77"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="291.96"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="304.72"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m294 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m300.67 103.33v-13.333" />
        <path d="m307.33 103.33v-13.333" />
        <path d="m314 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m320.67 110h6.6667v-6.6667h6.6667" />
        <path d="m320.67 116.67h13.333" />
        <path d="m320.67 123.33h13.333" />
        <path d="m320.67 130h6.6667v6.6667h6.6667" />
        <path d="m287.33 110h-6.6667v-6.6667h-6.6667" />
        <path d="m287.33 116.67h-13.333" />
        <path d="m287.33 123.33h-13.333" />
        <path d="m287.33 130h-6.6667v6.6667h-6.6667" />
        <path d="m294 136.67v6.6667h-6.6667v6.6667" />
        <path d="m300.67 136.67v13.333" />
        <path d="m307.33 136.67v13.333" />
        <path d="m314 136.67v6.6667h6.6667v6.6667" />
      </g>
      <g>
        <rect
          x="398.77"
          y="104.08"
          width="31.899"
          height="31.898"
          ry="0"
          stroke-width=".66666"
        />
        <rect
          x="401.96"
          y="107.27"
          width="25.518"
          height="25.518"
          ry="0"
          stroke-width=".66666"
        />
        <circle
          cx="414.72"
          cy="120.02"
          r="4.8473"
          fill-rule="evenodd"
          stroke-linejoin="round"
          stroke-width=".36861"
        />
      </g>
      <g stroke-width=".66666px">
        <path d="m404 103.33v-6.6667h-6.6667v-6.6667" />
        <path d="m410.67 103.33v-13.333" />
        <path d="m417.33 103.33v-13.333" />
        <path d="m424 103.33v-6.6667h6.6667v-6.6667" />
        <path d="m430.67 110h6.6667v-6.6667h6.6667" />
        <path d="m430.67 116.67h13.333" />
        <path d="m430.67 123.33h13.333" />
        <path d="m430.67 130h6.6667v6.6667h6.6667" />
        <path d="m397.33 110h-6.6667v-6.6667h-6.6667" />
        <path d="m397.33 116.67h-13.333" />
        <path d="m397.33 123.33h-13.333" />
        <path d="m397.33 130h-6.6667v6.6667h-6.6667" />
        <path d="m404 136.67v6.6667h-6.6667v6.6667" />
        <path d="m410.67 136.67v13.333" />
        <path d="m417.33 136.67v13.333" />
        <path d="m424 136.67v6.6667h6.6667v6.6667" />
      </g>
    </g>
    <g id="envs" fill="none">
      <rect
        x="55.297"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="60.248"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m65.198 11.248v49.505" />
        <path d="m70.149 11.248v49.505" />
        <path d="m75.099 11.248v49.505" />
        <path d="m80.05 11.248v49.505" />
        <path d="m85 11.248v49.505" />
        <path d="m89.95 11.248v49.505" />
        <path d="m94.901 11.248v49.505" />
        <path d="m99.851 11.248v49.505" />
        <path d="m104.8 11.248v49.505" />
        <path d="m60.248 16.198h49.505" />
        <path d="m60.248 21.149h49.505" />
        <path d="m60.248 26.099h49.505" />
        <path d="m60.248 31.05h49.505" />
        <path d="m60.248 36h49.505" />
        <path d="m60.248 40.95h49.505" />
        <path d="m60.248 45.901h49.505" />
        <path d="m60.248 50.851h49.505" />
        <path d="m60.248 55.802h49.505" />
      </g>
      <rect
        x="165.3"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="170.25"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m175.2 11.248v49.505" />
        <path d="m180.15 11.248v49.505" />
        <path d="m185.1 11.248v49.505" />
        <path d="m190.05 11.248v49.505" />
        <path d="m195 11.248v49.505" />
        <path d="m199.95 11.248v49.505" />
        <path d="m204.9 11.248v49.505" />
        <path d="m209.85 11.248v49.505" />
        <path d="m214.8 11.248v49.505" />
        <path d="m170.25 16.198h49.505" />
        <path d="m170.25 21.149h49.505" />
        <path d="m170.25 26.099h49.505" />
        <path d="m170.25 31.05h49.505" />
        <path d="m170.25 36h49.505" />
        <path d="m170.25 40.95h49.505" />
        <path d="m170.25 45.901h49.505" />
        <path d="m170.25 50.851h49.505" />
        <path d="m170.25 55.802h49.505" />
      </g>
      <rect
        x="275.3"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="280.25"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m285.2 11.248v49.505" />
        <path d="m290.15 11.248v49.505" />
        <path d="m295.1 11.248v49.505" />
        <path d="m300.05 11.248v49.505" />
        <path d="m305 11.248v49.505" />
        <path d="m309.95 11.248v49.505" />
        <path d="m314.9 11.248v49.505" />
        <path d="m319.85 11.248v49.505" />
        <path d="m324.8 11.248v49.505" />
        <path d="m280.25 16.198h49.505" />
        <path d="m280.25 21.149h49.505" />
        <path d="m280.25 26.099h49.505" />
        <path d="m280.25 31.05h49.505" />
        <path d="m280.25 36h49.505" />
        <path d="m280.25 40.95h49.505" />
        <path d="m280.25 45.901h49.505" />
        <path d="m280.25 50.851h49.505" />
        <path d="m280.25 55.802h49.505" />
      </g>
      <rect
        x="385.3"
        y="6.297"
        width="59.406"
        height="59.406"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".59406"
      />
      <rect
        x="390.25"
        y="11.248"
        width="49.505"
        height="49.505"
        opacity=".999"
        stroke-linejoin="round"
        stroke-width=".49505"
      />
      <g stroke-width=".49505px">
        <path d="m395.2 11.248v49.505" />
        <path d="m400.15 11.248v49.505" />
        <path d="m405.1 11.248v49.505" />
        <path d="m410.05 11.248v49.505" />
        <path d="m415 11.248v49.505" />
        <path d="m419.95 11.248v49.505" />
        <path d="m424.9 11.248v49.505" />
        <path d="m429.85 11.248v49.505" />
        <path d="m434.8 11.248v49.505" />
        <path d="m390.25 16.198h49.505" />
        <path d="m390.25 21.149h49.505" />
        <path d="m390.25 26.099h49.505" />
        <path d="m390.25 31.05h49.505" />
        <path d="m390.25 36h49.505" />
        <path d="m390.25 40.95h49.505" />
        <path d="m390.25 45.901h49.505" />
        <path d="m390.25 50.851h49.505" />
        <path d="m390.25 55.802h49.505" />
      </g>
    </g>
    <g id="agent-coordinator" fill="none">
      <path
        d="m287.22 208.25c0-9.999 0-19.999 3.3339-24.999s10-5.0006 13.396-8.9298c3.396-3.9292 3.5221-11.788 3.6482-19.649"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
        stroke-dasharray="1, 1"
      />
      <path
        d="m332.22 237.21c21.001-0.51956 42.002-1.0391 57.096-2.5812 15.094-1.5421 24.281-4.1065 28.686-10.062 4.4053-5.9552 4.0289-15.302 3.8407-26.046s-0.18823-22.885-0.18823-35.027"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
        stroke-dasharray="1.13133, 1.13133"
        stroke-width="1.1313"
      />
      <path
        d="m167.71 237.11c-22.309-0.51849-44.618-1.037-60.652-2.5759-16.034-1.5389-25.793-4.098-30.472-10.041-4.6796-5.9429-4.2798-15.27-4.0798-25.992 0.19995-10.722 0.19995-22.838 0.19995-34.955"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
        stroke-dasharray="1.16481, 1.16481"
        stroke-width="1.1648"
      />
      <path
        d="m212.71 208.25c0-9.999 0-19.999-3.3339-24.999s-10-5.0006-13.396-8.9298c-3.396-3.9292-3.5221-11.788-3.6482-19.649"
        marker-end="url(#marker27437)"
        marker-start="url(#Arrow1Mstart)"
        stroke-dasharray="1, 1"
      />
    </g>
    <g fill="none" stroke-dasharray="1, 1">
      <path
        id="coordinator-value"
        d="m230 300v70h-80"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
      <path
        id="coordinator-policy"
        d="m270 300v70h80"
        marker-end="url(#marker3806)"
        marker-start="url(#marker3806)"
      />
    </g>
    <g fill="none">
      <rect
        id="coordinator"
        x="170.63"
        y="220.13"
        width="158.74"
        height="59.743"
        stroke-width="1.2572"
      />
      <g>
        <rect x="180.5" y="230" width="10" height="40" />
        <rect x="223.5" y="230" width="10" height="40" />
        <rect x="266.5" y="230" width="10" height="40" />
        <rect x="309.5" y="230.5" width="10" height="40" />
      </g>
    </g>
  </g>
</svg>
<p>
  The improvements to performance that were made with the asynchronous advantage
  actor critic (A3C) were not necessarily due to the asynchronous aspect of the
  algorithm. Soon after the release of the algorithm researchers found out that
  the performance boost comes from the experiences that are generated by
  different environments and not the asynchronous updates that were made with
  Hogwild. In A2C (advantage actor critic) several environments are generated
  and a batch of experiences is gathered. But the update is done centrally by a
  coordinating entity. The advantage of this approach is the easier
  implementation and debugging, while keeping the decorelated experience tuples
  that result from the many distinct environments. Additionally as the update is
  done with experiences that can be packed into a batch, GPUs can be heavily
  used with A2C to improve the performance.
</p>
<p>
  There is no single paper that can be pointed to for the A2C algorithm. But the
  name has become popular after OpenAI released its baseline implementations for
  most common state of the art reinforcement learning algorithms.
</p>
<div class="separator" />

<style>
  svg {
    max-width: 700px;
  }
</style>
