<script>
  import Question from "$lib/Question.svelte";
  import Math from "$lib/Math.svelte";
  import Algorithm from "$lib/algorithm/Algorithm.svelte";
  import AlgorithmState from "$lib/algorithm/AlgorithmState.svelte";
  import AlgorithmForAll from "$lib/algorithm/AlgorithmForAll.svelte";
  import AlgorithmRepeat from "$lib/algorithm/AlgorithmRepeat.svelte";
</script>

<h1>Temporal Difference Learning</h1>
<Question
  >How can we use temporal-difference learning to solve tabular reinforcement
  learning problems.</Question
>
<div class="separator" />

<p class="info">
  Whereas conventional prediction-learning methods assign credit by means of the
  difference between predicted and actual outcomes, the new methods assign
  credit by means of the difference between temporally successive predictions.
</p>

<p>
  Instead of waiting for the end of an episode to use the difference between the
  predicted value function and the actual return <Math
    latex={String.raw`G_t`}
  /> temporal difference methods calculate the difference between the value of the
  current state <Math latex={String.raw`s`} /> and the value of the next state <Math
    latex={String.raw`s'`}
  />. These methods utilize temporal (as in time) differences in values in their
  update steps. As we do not have access to actual values for the next state we
  have to use estimates for prediction and target values, we have to bootstrap.
  Using temporal difference methods means that the agent does not have to wait
  for the end of the episode to apply an update step, but can update the
  estimations of the state or action value function after a single step.
</p>

<h2>Generalalized Policy Iteration</h2>
<p>
  Temporal difference learning is also based on generalized policy iteration and
  we are going to cover prediction and improvement in two separate steps.
</p>

<h3>Temporal Difference Prediction</h3>
<h4>Theory</h4>
<p>
  To solve the prediction problem means finding the true value function for a
  given policy <Math latex={String.raw`\pi`} />. The value function takes a
  state <Math latex={String.raw`s`} /> as an input and calculates the expected reward.
  Mathematically this can be expressed as follows.
</p>
<Math latex={String.raw`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`} />

<p>
  If we use the Bellman equation we can rewrite the definition of the value
  function in terms of the value of the next state.
</p>
<Math
  latex={String.raw`
    \begin{aligned}
    v_{\pi}(s) & = \mathbb{E_{\pi}}[G_t \mid S_t = s] \\
    & = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]
    \end{aligned}
  `}
/>
<p>
  Using the same logic the update step can be adjusted to reflect the recursive
  nature of the value function. The TD update rule uses bootsrapping, meaning
  that the target calculation is also based on an esimation.
</p>
<p>Monte Carlo Update:</p>
<Math latex={String.raw`V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]`} />
<p>Temporal Difference Update:</p>
<Math
  latex={String.raw`V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]`}
/>

<p>
  Due to recursive notation the target of the update rule, <Math
    latex={String.raw`R_{t+1} + \gamma v_{\pi}(S_{t+1})`}
  />, can be calculated at each single step. That means that it is not necessary
  to wait until the episode finishes to improve the estimate and that temporal
  difference learning is suited for continuing tasks.
</p>

<p class="info">
  The TD-Error quantifies the difference between the bootstrapped target and the
  estimation.
  <br />
  <Math
    latex={String.raw`\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)`}
  />
</p>

<h4>Algorithm</h4>
<p>
  Compared to Monte Carlo prediction the TD prediction algorithm looks cleaner.
  The interaction step and the update step are consolidated and do not have to
  be put into different loops or functions.
</p>

<Algorithm algoName={"Temporal Difference Prediction"}>
  <AlgorithmState
    >Input: environment <Math latex={`env`} />, policy <Math
      latex={String.raw`\mu`}
    />, state set <Math latex={String.raw`\mathcal{S}`} />, number of episodes <Math
      latex={String.raw`N`}
    /> , learning rate <Math latex={String.raw`\alpha`} />, discount factor <Math
      latex={String.raw`\gamma`}
    /></AlgorithmState
  >
  <AlgorithmState
    >Initialize: <Math latex={String.raw`V(s)`} /> for all <Math
      latex={String.raw`s \in \mathcal{S}`}
    />
    with zeros</AlgorithmState
  >
  <AlgorithmForAll>
    <span slot="condition">episodes <Math latex={String.raw`\in N`} /></span>
    <AlgorithmRepeat>
      <span slot="condition">state is terminal</span>
      <AlgorithmState
        >Generate a new action <Math latex={String.raw`a = \mu(s_t)`} />
      </AlgorithmState>
      <AlgorithmState
        >Generate a new state and reward <Math
          latex={String.raw`s_{t+1}, r_{t+1}=env(s_t, a_t)`}
        /></AlgorithmState
      >
      <AlgorithmState
        ><Math
          latex={String.raw`V(S) = V(S) + \alpha [R + \gamma V(S') - V(S)]`}
        /></AlgorithmState
      >
    </AlgorithmRepeat>
  </AlgorithmForAll>
  <AlgorithmState
    >Output: value function <Math latex={String.raw`V(s)`} /></AlgorithmState
  >
</Algorithm>
<h4>Implementation</h4>
<p>Coming soon...</p>

<h3>Temporal Difference Control</h3>
<p>
  SARSA is the On-Policy TD control algorithm. The same policy that is used to
  generate actions is also the one that is being improved.
</p>
<h4>SARSA</h4>

<p>
  SARSA is the On-Policy TD control algorithm. The same policy that is used to
  generate actions is also the one that is being improved.
</p>
<Math
  latex={String.raw`Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]`}
/>
<Algorithm algoName={"SARSA"}>
  <AlgorithmState
    >Input: environment <Math latex={`env`} />, state set <Math
      latex={String.raw`\mathcal{S}`}
    />, action set<Math latex={String.raw`\mathcal{A}`} />, number of episodes <Math
      latex={String.raw`N`}
    /> , learning rate <Math latex={String.raw`\alpha`} />, discount factor <Math
      latex={String.raw`\gamma`}
    />, epsilon <Math latex={String.raw`\epsilon`} />
  </AlgorithmState>
  <AlgorithmState
    >Initialize: <Math latex={String.raw`Q(s, a)`} /> for all <Math
      latex={String.raw`s \in \mathcal{S}`}
    />
    and
    <Math latex={String.raw`a \in \mathcal{A}`} />
    with zeros, policy <Math latex={String.raw`\mu(a \mid s)`} /> for all <Math
      latex={String.raw`a \in \mathcal{A}`}
    /> where <Math latex={String.raw`\mu`} /> is <Math
      latex={String.raw`\epsilon`}
    />-greedy
  </AlgorithmState>
  <AlgorithmForAll>
    <span slot="condition">episodes <Math latex={String.raw`\in N`} /></span>
    <AlgorithmState
      >Reset state <Math latex={String.raw`S`} /> and action <Math
        latex={String.raw`A`}
      /></AlgorithmState
    >
    <AlgorithmRepeat>
      <span slot="condition">state is terminal</span>
      <AlgorithmState
        >Generate tuple <Math latex={String.raw`(R,S',A')`} /> using policy <Math
          latex={String.raw`\mu`}
        /> and MDP <Math latex={String.raw`env`} />
      </AlgorithmState>
      <AlgorithmState
        ><Math
          latex={String.raw`Q(S, A) = Q(S) + \alpha [R + \gamma Q(S',A') - Q(S,A)]`}
        /></AlgorithmState
      >
      <AlgorithmState
        ><Math
          latex={String.raw`S \leftarrow S', A \leftarrow A'`}
        /></AlgorithmState
      >
    </AlgorithmRepeat>
  </AlgorithmForAll>
  <AlgorithmState
    >Output: value function <Math latex={String.raw`V(s)`} /></AlgorithmState
  >
</Algorithm>
<h4>Q-Learning</h4>

<p>
  Q-Learning is the On-Policy TD control algorithm. A different policy that is
  used to generate actions is being improved. Theoretically the algorithm should
  be able to learn the optimal control policy from a purely random
  action-selection policy.
</p>
<Math
  latex={String.raw`Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]`}
/>
<Algorithm algoName={"Q-Learning"}>
  <AlgorithmState
    >Input: environment <Math latex={`env`} />, state set <Math
      latex={String.raw`\mathcal{S}`}
    />, action set<Math latex={String.raw`\mathcal{A}`} />, number of episodes <Math
      latex={String.raw`N`}
    /> , learning rate <Math latex={String.raw`\alpha`} />, discount factor <Math
      latex={String.raw`\gamma`}
    />, epsilon <Math latex={String.raw`\epsilon`} />
  </AlgorithmState>
  <AlgorithmState
    >Initialize: <Math latex={String.raw`Q(s, a)`} /> for all <Math
      latex={String.raw`s \in \mathcal{S}`}
    />
    and
    <Math latex={String.raw`a \in \mathcal{A}`} />
    with zeros, policy <Math latex={String.raw`\mu(a \mid s)`} /> for all <Math
      latex={String.raw`a \in \mathcal{A}`}
    /> where <Math latex={String.raw`\mu`} /> is <Math
      latex={String.raw`\epsilon`}
    />-greedy
  </AlgorithmState>
  <AlgorithmForAll>
    <span slot="condition">episodes <Math latex={String.raw`\in N`} /></span>
    <AlgorithmState>Reset state <Math latex={String.raw`S`} /></AlgorithmState>
    <AlgorithmRepeat>
      <span slot="condition">state is terminal</span>
      <AlgorithmState
        >Generate tuple <Math latex={String.raw`(R,S',A)`} /> using policy <Math
          latex={String.raw`\mu`}
        /> and MDP <Math latex={String.raw`env`} />
      </AlgorithmState>
      <AlgorithmState
        ><Math
          latex={String.raw`Q(S, A) = Q(S) + \alpha [R + \gamma \max_aQ(S',a) - Q(S,A)]`}
        /></AlgorithmState
      >
      <AlgorithmState
        ><Math latex={String.raw`S \leftarrow S'`} /></AlgorithmState
      >
    </AlgorithmRepeat>
  </AlgorithmForAll>
  <AlgorithmState
    >Output: value function <Math latex={String.raw`V(s)`} /></AlgorithmState
  >
</Algorithm>
