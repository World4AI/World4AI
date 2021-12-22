<script>
  import Question from "$lib/Question.svelte";
  import Math from "$lib/Math.svelte";
  import Algorithm from "$lib/algorithm/Algorithm.svelte";
  import AlgorithmState from "$lib/algorithm/AlgorithmState.svelte";
  import AlgorithmForAll from "$lib/algorithm/AlgorithmForAll.svelte";
  import AlgorithmRepeat from "$lib/algorithm/AlgorithmRepeat.svelte";
  import AlgorithmIf from "$lib/algorithm/AlgorithmIf.svelte";
</script>

<h1>Monte Carlo Methods</h1>
<Question
  >How can we use monte carlo methods to solve tabular reinforcement learning
  problesm?</Question
>
<div class="separator" />

<h2>Motivation</h2>
<p>
  If we look at any definition of Monte Carlo methods, there is a high chance
  that the definition contains random sampling.
</p>

<p class="info">
  Monte Carlo methods are a broad class of computational algorithms that rely on
  repeated random sampling to obtain numerical results.
</p>
<p>
  When we apply Monte Carlo methods to reinforcement learning we sample episode
  paths, also called trajectories. The agent interacts with the environment and
  collects experience tuples that consist of states, actions and rewards.
</p>
<p>
  Monte Carlo methods are similar in spirit to bandit methods. The state-value
  and action-value functions can be estimated by taking the sampled trajectories
  and building averages. Unlike in bandits though, Monte Carlo methods are able
  to deal with environments where several non terminal states exist.
</p>
<p>
  Estimations can only be made once the trajectory is complete when the episode
  finishes, which means that Monte Carlo methods only work for episodic tasks.
</p>
<div class="separator" />

<h2>Generalized Policy Iteration</h2>
<p>
  The Monte Carlo algorithm will follow general policy iteration. We alternate
  between policy evaluation and policy improvement to find the optimal policy.
</p>
<h3>Policy Estimation</h3>

<h4>Theory</h4>
<p>
  Policy estimation deals with finding the true value function of a given policy
  <Math latex={String.raw`\pi`} />. Mathematically speaking we are looking for
  the expected sum of discounted rewards (also called returns) when the agent
  follows the policy <Math latex={String.raw`\pi`} />.
</p>
<Math latex={String.raw`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`} />

<p>
  A natural way to estimate the expected value of a random variable is to get
  samples from a distribution and to use the average as an estimate. In
  reinforcement learning the agent can estimate the expected value of returns
  for a policy <Math latex={String.raw`\pi`} /> by interacting with the environment,
  generating trajectories over and over again and building averages over the returns
  of the trajectories.
</p>

<p>
  Generally there are two methods to calculate the averages. Each time the agent
  faces a state during an episode is called a visit. In the “First Visit” Monte
  Carlo method only the return from the first visit to that state until the end
  of the episode is calculated. If the state is visited several times during an
  episode, the additional visits are not considered in the calculation. While in
  the “Every Visit” method each visit is counted. The “First Visit” method is
  more popular and generally more straightforward and is going to be covered in
  this section, but the algorithms can be easily adjusted to account for the
  “Every Visit” method.
</p>

<p>
  To make the calculations of the averages computationally efficient we are
  going to use the incremental implementation that we already used for n-armed
  bandits.
</p>
<Math
  latex={String.raw`NewEstimate \leftarrow OldEstimate + StepSize*[Target - OldEstimate]`}
/>

<h4>Algorithm</h4>
<p>The algorithm is divided into two steps.</p>
<ul>
  <li>
    The agent generates a trajectory using the policy <Math
      latex={String.raw`\pi`}
    />.
  </li>
  <li>
    The agent improves the estimation for the state value function <Math
      latex={String.raw`V(s)`}
    />. For that purpose the agent loops over the previously generated
    trajectory and for each experience tuple it determines if it deals with a
    first visit to that state <Math latex={`s`} />. If it does it calculates the
    discounted sum of rewards from that point on to the terminal state <Math
      latex={String.raw`G_{t:T} = \sum_{k=t}^T \gamma^{k-t}R_t`}
    />. Finally the agent performs an update step by using the incremental
    average calculation <Math
      latex={String.raw`V(s) = V(s) + \alpha [G_{t:T} - V(s)]`}
    />.
  </li>
</ul>

<Algorithm algoName={"Monte Carlo Prediction: First Visit"}>
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
    <AlgorithmState>(1) INTERACTION WITH THE ENVIRONMENT</AlgorithmState>
    <AlgorithmState>create trajectory as empty list [...]</AlgorithmState>
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
        >Push the tuple <Math latex={String.raw`(s_t, r_{t+1})`} /> into the trajectory
        list.
      </AlgorithmState>
    </AlgorithmRepeat>
    <AlgorithmState>(2) ESTIMATION OF THE VALUE FUNCTION</AlgorithmState>
    <AlgorithmState
      >Create Visited(s) = False for all <Math
        latex={String.raw`s \in \mathcal{S}`}
      /></AlgorithmState
    >
    <AlgorithmForAll>
      <span slot="condition">tuples in trajectory list</span>
      <AlgorithmState
        ><Math latex={String.raw`s \leftarrow`} /> state from tuple
      </AlgorithmState>
      <AlgorithmIf>
        <span slot="condition">Visited(s) is True</span>
        <AlgorithmState>Skip and go to next tuple</AlgorithmState>
      </AlgorithmIf>
      <AlgorithmIf>
        <span slot="condition">Visited(s) is False</span>
        <AlgorithmState>Visited(s) = True</AlgorithmState>
      </AlgorithmIf>
      <AlgorithmState
        ><Math
          latex={String.raw`G_{t:T} = \sum_{k=t}^T \gamma^{k-t}R_t`}
        /></AlgorithmState
      >
      <AlgorithmState
        ><Math
          latex={String.raw`V(s) = V(s) + \alpha [G_{t:T} - V(s)]`}
        /></AlgorithmState
      >
    </AlgorithmForAll>
  </AlgorithmForAll>
  <AlgorithmState>Output: value function V(s)</AlgorithmState>
</Algorithm>

<h4>Implementation</h4>
<p>Coming soon ....</p>

<h3>Policy Improvement And Control</h3>
<h4>Theory</h4>
<p>
  The value iteration algorithm that we applied in the dynamic programming
  section used the following update step.
</p>
<Math
  latex={String.raw`
    \begin{aligned}
    v_{k+1}(s) & \doteq \max_a \mathbb{E}[R_{t+1} + \gamma v_k (S_{t+1}) \mid S_t = s, A_t = a] \\
    & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]
    \end{aligned}
  `}
/>
<p>
  This exact update step is not going to work with Monte Carlo methods, because
  that would require the full knowledge of the model. We would have to know the
  transition probabilities from state <Math latex={String.raw`s`} /> to state <Math
    latex={String.raw`s'`}
  /> and the corresponding reward <Math latex={String.raw`r`} />.
</p>

<p>
  If we look closely at the above expression, we should notice that we can
  rewrite the update rule in terms of an action-value function.
</p>
<Math
  latex={String.raw`
    \begin{aligned}
    v_{k+1}(s) & \doteq \max_a \mathbb{E}[R_{t+1} + \gamma v_k (S_{t+1}) \mid S_t = s, A_t = a] \\
    & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')] \\
    & = \max_a q_k(s, a) \\
    \end{aligned}
  `}
/>
<p>
  With those rewrites we do not require the knowledge of the model, but it
  becomes obvious that the key is to estimate the action-value function and not
  the state-value function. Having an estimate of an action-value function
  allows the agent to select better actions by acting greedily and to gradually
  improve the policy towards the optimal policy. To estimate the action-value
  function we will still generate episodes and compute averages, but the
  averages are not going to be for a state, but for a state-action pair.
</p>
<p>
  There is still one problem that we face without the knowledge of the model of
  the MDP though. If our policy is fully deterministic and thus avoids some
  state-action pairs by design, then we can not compute a good estimate for
  certain state-action pairs and thus might not arrive at the optimal policy.
  The solution is to use an <Math latex={String.raw`\epsilon`} />-greedy policy,
  meaning that with a probability of <Math latex={String.raw`\epsilon`} /> we take
  a random action and with probability of <Math
    latex={String.raw`1-\epsilon`}
  /> we take the greedy action. That way we are guaranteed that all state-action
  pairs are going to be visited.
</p>
<p>
  Before we move on to the implementation of the Monte Carlo control algorithm
  it is important to discuss the difference between on-policy and off-policy
  methods. Once the need arises to explore the environment we could ask
  ourselves, “Do we need to improve the same policy that is used to generate
  actions or can we learn the optimal policy while using the data that was
  produced by a different policy?”. To frame the question differently “Is it
  possible to learn the optimal policy while only selecting random actions?”.
  That depends on the design of the algorithm. On-policy methods improve the
  same policy that is also used to generate the actions, while off-policy
  methods improve a policy that is not the one that is used to generate the
  trajectories. The algorithm that is covered below is an on-policy algorithm.
</p>

<h4>Algorithm</h4>
<Algorithm algoName={"Monte Carlo Control: First Visit"}>
  <AlgorithmState
    >Input: environment <Math latex={`env`} />, state set <Math
      latex={String.raw`\mathcal{S}`}
    />, action set <Math latex={String.raw`\mathcal{A}`} />, number of episodes <Math
      latex={String.raw`N`}
    /> , learning rate <Math latex={String.raw`\alpha`} />, discount factor <Math
      latex={String.raw`\gamma`}
    />, epsilon <Math latex={String.raw`\epsilon`} /></AlgorithmState
  >
  <AlgorithmState
    >Initialize: <Math latex={String.raw`Q(s, a)`} /> for all <Math
      latex={String.raw`s \in \mathcal{S}`}
    /> and <Math latex={String.raw`a \in \mathcal{A}`} />
    with zeros, policy <Math latex={String.raw`\mu(a \mid s)`} /> for all <Math
      latex={String.raw`a \in \mathcal{A}`}
    /> where <Math latex={String.raw`\mu`} /> is <Math
      latex={String.raw`\epsilon`}
    />-greedy</AlgorithmState
  >
  <AlgorithmForAll>
    <span slot="condition">episodes <Math latex={String.raw`\in N`} /></span>
    <AlgorithmState>(1) INTERACTION WITH THE ENVIRONMENT</AlgorithmState>
    <AlgorithmState>create trajectory as empty list [...]</AlgorithmState>
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
        >Push the tuple <Math latex={String.raw`(s_t, a_t, r_{t+1})`} /> into the
        trajectory list.
      </AlgorithmState>
    </AlgorithmRepeat>
    <AlgorithmState>(2) ESTIMATION OF THE VALUE FUNCTION</AlgorithmState>
    <AlgorithmState
      >Create Visited(s) = False for all <Math
        latex={String.raw`s \in \mathcal{S}`}
      /></AlgorithmState
    >
    <AlgorithmForAll>
      <span slot="condition">tuples in trajectory list</span>
      <AlgorithmState
        ><Math latex={String.raw`s \leftarrow`} /> state and <Math
          latex={String.raw`a \leftarrow`}
        /> action from tuple
      </AlgorithmState>
      <AlgorithmIf>
        <span slot="condition">Visited(s) is True</span>
        <AlgorithmState>Skip and go to next tuple</AlgorithmState>
      </AlgorithmIf>
      <AlgorithmIf>
        <span slot="condition">Visited(s) is False</span>
        <AlgorithmState>Visited(s) = True</AlgorithmState>
      </AlgorithmIf>
      <AlgorithmState
        ><Math
          latex={String.raw`G_{t:T} = \sum_{k=t}^T \gamma^{k-t}R_t`}
        /></AlgorithmState
      >
      <AlgorithmState
        ><Math
          latex={String.raw`Q(s, a) = Q(s, a) + \alpha [G_{t:T} - Q(s, a)]`}
        /></AlgorithmState
      >
    </AlgorithmForAll>
  </AlgorithmForAll>
  <AlgorithmState>Output: value function V(s)</AlgorithmState>
</Algorithm>

<h4>Implementation</h4>
<p>Coming Soon ...</p>
<div class="separator" />
<h2>Sources</h2>
<div class="separator" />
