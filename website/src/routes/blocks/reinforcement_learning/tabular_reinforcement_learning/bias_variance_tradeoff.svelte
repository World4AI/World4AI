<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Button from "$lib/Button.svelte";
  import Table from "$lib/Table.svelte";

  let biasHeader = ["Probability", "Value"];
  let biasData = [
    ["15%", 1],
    ["15%", 2],
    ["30%", 3],
    ["30%", 4],
  ];

  let varianceHeader1 = ["Probability", "Value"];
  let varianceData1 = [
    ["50%", -1],
    ["50%", 1],
  ];

  let varianceHeader2 = ["Probability", "Value"];
  let varianceData2 = [
    ["50%", -5],
    ["50%", 5],
  ];

  let height = 500;
  let width = 500;
  let outerRadii = [75, 55, 35, 15];
  let innerRadius = 2;

  let rectSize = width / 2;

  let coordinates = [
    {
      x: width / 4,
      y: width / 4,
      text: "Low Bias / Low Variance",
      bias: 0,
      variance: 5,
      data: [],
    },
    {
      x: width / 4,
      y: width / 2 + width / 4,
      text: "High Bias / Low Variance",
      bias: 50,
      variance: 5,
      data: [],
    },
    {
      x: width / 2 + width / 4,
      y: width / 4,
      text: "Low Bias / High Variance",
      bias: 0,
      variance: 35,
      data: [],
    },
    {
      x: width / 2 + width / 4,
      y: width / 2 + width / 4,
      text: "High Bias / High Variance",
      bias: 50,
      variance: 35,
      data: [],
    },
  ];

  //fill the data
  function fillData() {
    coordinates.forEach((coordinate) => {
      for (let i = 0; i < 10; i++) {
        let pointX =
          coordinate.x +
          coordinate.bias +
          Math.random() * (coordinate.variance * 2) -
          coordinate.variance;
        let pointY =
          coordinate.y +
          coordinate.bias +
          Math.random() * (coordinate.variance * 2) -
          coordinate.variance;
        let point = { x: pointX, y: pointY };

        coordinate.data = [...coordinate.data, point];
      }
    });
    coordinates = [...coordinates];
    console.log(coordinates);
  }
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Bias-Variance Tradeoff</title>
  <meta
    name="description"
    content="The bias-variance tradeoff plays a huge role in reinforcement learning. Monte Carlo methods show high variance and low bias, while temporal difference methods show high bias and low variance."
  />
</svelte:head>

<h1>Bias-Variance Tradeoff</h1>
<Question
  >What is the bias-variance tradeoff and how does it relate to Monte Carlo and
  TD learning?</Question
>
<div class="separator" />

<p>
  The bias-variance tradeoff plays a tremendous role in reinforcement learning
  and machine learning in general. Ideally we want to reduce both as much as
  possible, but reducing one means increasing the other at the same time. There
  is a tradeoff to be made. This tradeoff needs to be considered when choosing
  between Monte Carlo and temporal difference methods.
</p>
<div class="separator" />

<h2>Intuition</h2>
<p>
  Before we dive in into the mathematical foundations of the bias-variance
  tradeoff, we should try and get an intuitive understanding of the concept.
  Below is an interactive example that should facilitate the learning. The
  bull's eye at the center of each block represents the true expected value of
  some random variable. The closer our estimates are to the bull's eye the
  better the better. Click at the button a couple of times to get some samples
  of the estimator and try to get a feel for what variance and bias might mean.
</p>
<div class="svg-container">
  <svg viewBox="0 0 {width} {height}">
    <g class="boxes">
      {#each coordinates as coordinate}
        <text
          dominant-baseline="middle"
          text-anchor="middle"
          font-size="15px"
          x={coordinate.x}
          y={coordinate.y + rectSize / 2 - 15}
          fill="var(--text-color)"
          >{coordinate.text}
        </text>
        <rect
          stroke="none"
          x={coordinate.x - rectSize / 2}
          y={coordinate.y - rectSize / 2}
          width={rectSize}
          height={rectSize}
        />
        <circle
          class="inner-circle"
          cx={coordinate.x}
          cy={coordinate.y}
          r={innerRadius}
        />
        <!-- draw the outer circles -->
        {#each outerRadii as radius}
          <circle
            class="outer-circle"
            cx={coordinate.x}
            cy={coordinate.y}
            r={radius}
          />
        {/each}

        <!-- draw the data points -->
        {#each coordinate.data as dataPoint}
          <circle
            class="inner-circle"
            cx={dataPoint.x}
            cy={dataPoint.y}
            r={innerRadius}
          />
        {/each}
      {/each}
    </g>
  </svg>
  <div class="flex-center">
    <Button value={"Draw Samples"} on:click={fillData} />
  </div>
</div>
<div class="separator" />
<p>
  The top left corner represents a situation with low bias and low variance. The
  estimates of the random variable come on average very close to the bull's eye
  (low bias) and do not stray to much from each other (low variance).
</p>
<p>
  The top right corner represents a situation with low bias and high variance.
  The average of all estimates is still very close to the center of the target
  (low bias), yet the estimates are dispersed (high variance).
</p>
<p>
  The bottom left corner shows a high bias and low variance situation. The
  average of the samples of our estimate is far from the bull's eye (high bias),
  but the samples do not vary too much (low variance).
</p>
<p>
  Finally the bottom right corner depicts a high bias high variance situation.
</p>
<p>
  It is hopefully clear, that the situation in the top left corner is the most
  desirable one, but reality is rarely that convenient. Usually we have to
  choose between high variance and high bias and we need to find a balance
  between the two, a tradeoff.
</p>
<div class="separator" />

<h2>Bias</h2>
<p>
  In statiscics we define the bias of the estimator as the difference between
  the expected value of the estimator <Latex
    >{String.raw`\mathbb{E}[\hat{\theta}]`}</Latex
  >and the true parameter <Latex>{String.raw`\theta`}</Latex>.
</p>
<Latex
  >{String.raw`bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta`}</Latex
>
<p>
  The true parameter <Latex>{String.raw`\theta`}</Latex> that we are most interested
  in is the expected value <Latex>{String.raw`\mathbb{E}[X]`}</Latex>of some
  random variable <Latex>{String.raw`X`}</Latex>.
</p>
<Latex
  >{String.raw`bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \mathbb{E}[X]`}</Latex
>
<Table header={biasHeader} data={biasData} />
<p>
  In the table above we see the distribution of the random variable <Latex
    >{String.raw`X`}</Latex
  >. The calculation of the expectation is straightforward when we know the
  actual probability distribution.
</p>
<Latex
  >{String.raw`\mathbb{E}[X] = 0.15 * 1 + 0.15 * 2 + 0.3 * 3 + 0.3 * 4 = 2.55`}</Latex
>
<p>
  In reality we rarely know the true distribution of <Latex
    >{String.raw`X`}</Latex
  > and can not directly calculate the expected value, therefore we have to use some
  estimate
  <Latex>{String.raw`\hat{\theta} `}</Latex> as a proxy for the true <Latex
    >{String.raw`\mathbb{E}[X]`}</Latex
  >. The most straightforward way to estimate the expected value of a random
  variable is to draw samples from the distribution and to use the individual
  random values
  <Latex>{String.raw`X`}</Latex> as the estimate for <Latex
    >{String.raw`\hat{\theta}`}</Latex
  >. But is there any bias by using the random samples as an estimate? It turns
  out that using
  <Latex>{String.raw`X`}</Latex> as an estimate for <Latex
    >{String.raw`\mathbb{E}[X]`}</Latex
  > introduces no bias.
</p>
<Latex
  >{String.raw`
\begin{aligned}
  & \theta = \mathbb{E}[X] \\
  & \hat{\theta} = X \\
  & bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta = \mathbb{E}[X] - \mathbb{E}[X] = 0
\end{aligned}
  `}</Latex
>
<p>
  If on the other hand we used the number 3 as the estimate for the expected
  value we would introduce a bias.
</p>
<Latex
  >{String.raw`
\begin{aligned}
  & \theta = \mathbb{E}[X] \\
  & \hat{\theta} = 3 \\
  & bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta = \mathbb{E}[3] - \mathbb{E}[X] = 3 - 2.55 = 0.45
\end{aligned}
  `}</Latex
>
<p>
  But how do the discussions regarding bias extend to the choice between monte
  carlo methods and temporal difference methods? The example below might provide
  some answers.
</p>
<svg width="500" height="100" version="1.1" viewBox="0 0 500 100">
  <defs>
    <marker id="marker27437" overflow="visible" orient="auto">
      <path
        stroke="var(--text-color)"
        fill="var(--text-color)"
        transform="scale(-.8)"
        d="m8.7186 4.0337-10.926-4.0177 10.926-4.0177c-1.7455 2.3721-1.7354 5.6175-6e-7 8.0354z"
        fill-rule="evenodd"
        stroke-linejoin="round"
        stroke-width=".625"
      />
    </marker>
  </defs>
  <g transform="translate(-1.5 -325)">
    <rect
      x="28"
      y="350"
      width="50"
      height="50"
      fill="none"
      fill-rule="evenodd"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
    <rect x="128" y="350" width="50" height="50" />
    <rect x="228" y="350" width="50" height="50" />
    <rect x="328" y="350" width="50" height="50" />
    <rect
      x="425"
      y="350"
      width="50"
      height="50"
      fill-rule="evenodd"
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="1"
    />
    <g fill="none" stroke="var(--text-color)" stroke-width="1px">
      <path d="m84 376h40" marker-end="url(#marker27437)" />
      <path d="m183.34 376h40" marker-end="url(#marker27437)" />
      <path d="m283.34 376h40" marker-end="url(#marker27437)" />
      <path d="m381.34 376h40" marker-end="url(#marker27437)" />
    </g>
    <g fill="var(--text-color)" font-size="35px">
      <text x="36.726562" y="388" style="line-height:1.25"
        ><tspan x="36.726562" y="388">-1</tspan></text
      >
      <text x="144.72656" y="388" style="line-height:1.25"
        ><tspan x="144.72656" y="388">-1</tspan></text
      >
      <text x="244.72656" y="388" style="line-height:1.25"
        ><tspan x="244.72656" y="388">-1</tspan></text
      >
      <text x="344.72656" y="388" style="line-height:1.25"
        ><tspan x="344.72656" y="388">-1</tspan></text
      >
      <text x="444.72656" y="388" style="line-height:1.25"
        ><tspan x="444.72656" y="388">1</tspan></text
      >
    </g>
  </g>
</svg>
<p>
  Imagine an episodic environment where the agent starts at the left box and has
  to arrive at the right box. Each step that the agent takes generates a
  negative reward of -1. Only the terminal state provides a positive reward. To
  make the calculations simple we make the environment and the policy of the
  agent fully deterministic. In our example the agent follows the policy of
  always going right and the environment always transitions in the desired way.
  We also do not use any discounting. Following these assumptions the agent
  receives a return <Latex>G_t</Latex> of -2 at each single episode when the agent
  starts at the left state, which we designate as state 0.
</p>
<p>
  When we use policy evaluation our goal is to find the correct value function <Latex
    >{String.raw`v_{\pi}(s)`}</Latex
  >, which is the expected value of the rewards <Latex>{String.raw`G_t`}</Latex>
  when following the policy <Latex>{String.raw`\pi`}</Latex>. It turns out that
  Monte Carlo methods are not biased, because we use the full returns
  <Latex>{String.raw`G_t`}</Latex> as an estimator of the value function.
</p>
<Latex
  >{String.raw`
    \begin{aligned}
    & \theta = \mathbb{E}[G_0] \\
    & \hat{\theta} = G_0 \\
    & bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta = \mathbb{E}[G_0] - \mathbb{E}[G_0] = -2 - (-2) = 0 
    \end{aligned}
`}</Latex
>
<p>
  The return is an unbiased estimator of the value function and that would not
  change even if the environment and the policy of the agent were stochastic.
</p>
<p>
  This is not the case for temporal difference methods. Let us assume that we
  set the initial values of the state value function to 0, <Latex
    >{String.raw`v_{\pi}(s) = 0`}</Latex
  >. TD methods use bootstrapping, therefore the estimate of the value for the
  0th state is <Latex>{String.raw`R_t + v_{\pi}(1) = -1 + 0 = -1`}</Latex>,
  which makes bootstrapping biased, <Latex
    >{String.raw`\mathbb{E}[R_t + v_{\pi}(S_{t+1})] \neq \mathbb{E}[G_t]`}</Latex
  >. The reason we can get away with this type of bias in temporal difference
  methods is the continuous improvement of the estimate of the value function,
  which reduces the bias over time.
</p>
<div class="separator" />

<h2>Variance</h2>
<p>
  Intuitively the variance of an estimator tells us how strongly the estimator
  fluctuates around the expected value of an estimator. Mathematically we can
  express the idea in the following way.
</p>
<Latex
  >{String.raw`var(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]`}</Latex
>
<p>
  Let's assume we face the below distribution and decide that we would like to
  use the random variable <Latex>X</Latex> as the estimator for the expected value
  <Latex>{String.raw`\mathbb{E}[X]`}</Latex>. The true expected value of the
  estimate is 0, but the individual draws from the distribution can vary around
  the expected value, the variance is not 0.
</p>
<Table data={varianceData1} header={varianceHeader1} />
<Latex
  >{String.raw`
\begin{aligned}
  var(\hat{\theta}) & = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2] \\
  & = \mathbb{E}[(\hat{\theta} - 0)^2] \\
  & = 0.5 [(1 - 0)^2] +  0.5 [(-1 - 0)^2] = 1
\end{aligned}
  `}</Latex
>
<p>
  The calculation looks different if we use the random variable <Latex>X</Latex>
  as the estimate of the expected value of the below distribution. The expected value
  is the same, but the variance as you can imagine is higher.
</p>
<Table data={varianceData2} header={varianceHeader2} />
<Latex
  >{String.raw`
\begin{aligned}
  var(\hat{\theta}) & = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2] \\
  & = \mathbb{E}[(\hat{\theta} - 0)^2] \\
  & = 0.5 [(5 - 0)^2] +  0.5 [(-5 - 0)^2] \\
  & = 0.5 * 25 +  0.5 * 25 = 25
\end{aligned}
  `}</Latex
>
<p>
  Let us build some intuition regarding the variance in Monte Carlo and temporal
  difference methods.
</p>
<p>
  Considering that the environment is usually stochastic and that the rules of
  the MPD are often complex, you generally require a lot of samples to
  approximate the expected value of the returns. When we use Monte Carlo methods
  in reinforcement learning the randomness accumulates through many actions and
  transitions, thus the trajectories might vary.
</p>
<p>
  The randomness in temporal difference methods on the other hand is manageable.
  We take a single action and add the result to the state or action value of the
  next state. The randomness is therefore only present during a single state.
  The rest of the randomness is already incorporated in the state or action
  value function.
</p>
<p>
  Monte Carlo methods have more variance and temporal difference methods have
  more bias. In practice we usually look for some compromise between the two
  extremes.
</p>
<div class="separator" />

<style>
  .svg-container {
    max-width: 500px;
  }
  rect {
    stroke: var(--text-color);
    fill: none;
  }
  .inner-circle {
    fill: var(--text-color);
    stroke: var(--text-color);
  }
  .outer-circle {
    fill: none;
    stroke: var(--text-color);
  }
</style>
