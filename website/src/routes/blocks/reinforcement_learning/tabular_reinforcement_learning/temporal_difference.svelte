<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Temporal Difference (TD) Learning</title
  >
  <meta
    name="description"
    content="Temporal Difference methods use bootstrapping for reinforcement learning prediction and control tasks."
  />
</svelte:head>

<h1>Temporal Difference Learning</h1>
<Question
  >How can we use temporal difference learning to solve tabular reinforcement
  learning problems?</Question
>
<div class="separator" />

<p class="info">
  Whereas conventional prediction-learning methods assign credit by means of the
  difference between predicted and actual outcomes, the new methods assign
  credit by means of the difference between temporally successive predictions.
</p>
<p>
  Instead of waiting for the end of an episode to use the difference between the
  predicted state or action value and the actual return <Latex>G_t</Latex>
  temporal difference methods calculate the difference between the value of the current
  state <Latex>s</Latex> and the value of the next state <Latex>s'</Latex>.
</p>
<div class="svg-container">
  <svg version="1.1" viewBox="0 0 500 150" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="marker4005" overflow="visible" orient="auto">
        <path
          transform="scale(.8)"
          d="m5.77 0-8.65 5v-10z"
          fill-rule="evenodd"
          stroke="var(--text-color)"
          stroke-width="1px"
          fill="var(--text-color)"
        />
      </marker>
      <marker id="marker9686" overflow="visible" orient="auto">
        <path
          transform="scale(.4)"
          d="m-5-5v10h10v-10z"
          fill="var(--text-color)"
          fill-rule="evenodd"
          stroke="var(--text-color)"
          stroke-width="1px"
        />
      </marker>
    </defs>
    <g stroke="var(--text-color)">
      <g fill="none">
        <rect
          id="final-state"
          x="441.05"
          y="66.728"
          width="14.142"
          height="14.142"
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="0.5px"
        />
        <g stroke-width="1px">
          <path d="m52.388 74.05h36.223" marker-end="url(#marker4005)" />
          <path d="m115.32 74.05h36.223" marker-end="url(#marker4005)" />
          <path d="m190.27 74.05h36.223" marker-end="url(#marker4005)" />
        </g>
      </g>
      <g
        id="states"
        fill="none"
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-width="0.8px"
      >
        <circle cx="39.568" cy="76.49" r="10" />
        <circle cx="172.5" cy="76.49" r="10" fill-rule="evenodd" />
        <circle cx="311.8" cy="76.49" r="10" fill-rule="evenodd" />
      </g>
      <path
        d="m254.62 74.05h36.223"
        fill="none"
        marker-end="url(#marker4005)"
        stroke-width="1px"
      />
      <g
        id="actions"
        fill="var(--text-color)"
        fill-rule="evenodd"
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-width="0.5px"
        stroke="black"
      >
        <circle cx="103.52" cy="74.05" r="4.9497" />
        <circle cx="243.41" cy="74.05" r="4.9497" />
        <circle cx="382" cy="74.05" r="4.9497" />
      </g>
      <g fill="none">
        <g stroke-width="0.7px">
          <path d="m328.87 74.05h36.223" marker-end="url(#marker4005)" />
          <path d="m391.8 73.766h36.223" marker-end="url(#marker4005)" />
          <path
            d="m36.453 49.854v-20.631h133.57v20.631"
            marker-end="url(#marker4005)"
            marker-mid="url(#marker9686)"
            marker-start="url(#marker9686)"
          />
        </g>
        <path d="m20 38.462v-24.462h170v79.5h-170z" stroke-dasharray="4, 2" />
        <path d="m28 110 10 20 10-20" />
        <path d="m162.66 110 10 20 10-20" />
        <path d="m60 120h90" stroke-dasharray="2, 4" />
      </g>
    </g>
  </svg>
</div>
<p>
  These methods utilize temporal (as in time) differences in values in their
  update steps. As we do not have access to actual values for the next state we
  have to use estimates for target values, we need to use bootstrap. When the
  agent uses temporal difference methods the agent does not have to wait for the
  end of the episode to apply an update to the estimate of the value function,
  but can instead update the estimation function after a single step.
</p>
<p>
  Temporal difference learning is also based on generalized policy iteration,
  which alternates between policy evaluation and policy improvement to arrive at
  the optimal policy.
</p>

<div class="separator" />
<h2>Prediction</h2>
<p>
  To solve the prediction problem means finding the true value function <Latex
    >{String.raw`v_\pi(s)`}</Latex
  > for a given policy <Latex>{String.raw`\pi`}</Latex>. We can express this
  task either using the definition of returns <Latex>G_t</Latex>.
</p>
<Latex>{String.raw`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`}</Latex>
<p>
  Or we we can use the Bellman equation to rewrite the definition of the value
  function recursively in terms of the value of the next state.
</p>
<Latex
  >{String.raw`
    v_{\pi}(s) = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]
  `}</Latex
>
<p>
  Using the same logic the update step can be adjusted to reflect the recursive
  nature of the value function by using bootstrapping.
</p>
<Highlight>
  <Latex
    >{String.raw`\text{MC: }V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]`}</Latex
  >
</Highlight>
<Highlight>
  <Latex
    >{String.raw`{TD: }V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]`}</Latex
  >
</Highlight>
<p>
  Due to bootstrapping the target <Latex
    >{String.raw`R_{t+1} + \gamma v_{\pi}(S_{t+1})`}</Latex
  >
  can be calculated at each single step. That means that it is not necessary to wait
  until the episode finishes to improve the estimate and that temporal difference
  learning is suited for continuing tasks.
</p>
<Latex
  >{String.raw`V(S_t) \leftarrow V(S_t) + \alpha [\underbrace{R_{t+1} + \overbrace{\gamma V(S_{t+1}) - V(S_t)}^{\boxed{\text{Temporal Difference}}}}_{\boxed{\delta \text{: TD Error}}}]`}</Latex
>
<p>
  The quantity <Latex>{String.raw`V(S_{t+1}) - V(S_t)`}</Latex> is the temporal difference:
  difference of values measured at successive states. The quantity <Latex
    >{String.raw`R_{t+1} + \gamma V(S_{t+1}) - V(S_t)`}</Latex
  > is called TD error. TD error the difference between the bootstrapped target and
  the estimated value. The larger the error
  <Latex>\delta</Latex>, the larger the adjustment that is done to the estimate.
</p>
<p>
  The practical implementation once again relies on a value function contained
  in a table that maps states <Latex>s</Latex> to values <Latex>V(s)</Latex>.
  Unlike in Monte Carlo methods in TD methods the agent updates the value <Latex
    >V(s)</Latex
  > for the encountered state <Latex>s</Latex> after each single step of the interaction.
</p>
<p>
  Compared to Monte Carlo prediction the TD prediction implementation looks
  cleaner. The interaction step and the update step are consolidated and do not
  have to be put into different loops or functions.
</p>

<div class="separator" />
<h2>Control</h2>
<p>
  To find the optimal policy we need to estimate the action value function <Latex
    >Q(s,a)</Latex
  >. Below we are going to discuss SARSA, the on-policy control algorithm, and
  Q-learning, the off-policy control algorithm.
</p>
<h3>SARSA</h3>
<p>
  SARSA is the on-policy TD control algorithm. The same policy that is used to
  generate actions (<Latex>\epsilon</Latex>-greedy in our case) is also the one
  that is being improved. At each timestep the agent improves the action-value
  function, by using a bootstrapped version of the target.
</p>
<Highlight>
  <Latex
    >{String.raw`Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]`}</Latex
  >
</Highlight>
<p>
  Both actions <Latex>A_t</Latex> and <Latex>{String.raw`A_{t+1}`}</Latex> are picked
  using the <Latex>\epsilon</Latex>-greedy policy <Latex>\pi(a \mid s)</Latex>.
</p>
<p>
  The name SARSA can be easily understood by writing out those states, actions
  and the reward that are needed to apply one update step: <Latex
    >{String.raw`[S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}]`}</Latex
  >
  .
</p>

<h3>Q-Learning</h3>
<p>
  Q-Learning is the off-policy TD control algorithm. In this algorithm we use <Latex
    >\epsilon</Latex
  >-greedy to take actions, yet a purely greedy strategy is the one that we aim
  to improve.
</p>
<Highlight>
  <Latex
    >{String.raw`Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]`}</Latex
  >
</Highlight>
<p>
  Unlike in SARSA, the bootstrapped value from the target is caluclated using
  the max operator: <Latex>{String.raw`\max_a Q(S_{t+1},a)`}</Latex>.
</p>
<div class="separator" />

<style>
  .svg-container {
    max-width: 800px;
  }
</style>
