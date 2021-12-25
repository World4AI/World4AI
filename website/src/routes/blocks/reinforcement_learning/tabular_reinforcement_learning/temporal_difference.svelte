<script>
  import Question from "$lib/Question.svelte";
  import Code from "$lib/Code.svelte";
  import Latex from "$lib/Latex.svelte";
  import Algorithm from "$lib/algorithm/Algorithm.svelte";
  import AlgorithmState from "$lib/algorithm/AlgorithmState.svelte";
  import AlgorithmForAll from "$lib/algorithm/AlgorithmForAll.svelte";
  import AlgorithmRepeat from "$lib/algorithm/AlgorithmRepeat.svelte";
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
  predicted value function and the actual return <Latex>G_t</Latex>
  temporal difference methods calculate the difference between the value of the current
  state <Latex>s</Latex> and the value of the next state <Latex>s'</Latex>.
  These methods utilize temporal (as in time) differences in values in their
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
  given policy <Latex>{String.raw`\pi`}</Latex>. The value function takes a
  state <Latex>s</Latex> as an input and calculates the expected reward. Mathematically
  this can be expressed as follows.
</p>
<Latex>{String.raw`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`}</Latex>
<p>
  If we use the Bellman equation we can rewrite the definition of the value
  function in terms of the value of the next state.
</p>
<Latex
  >{String.raw`
    \begin{aligned}
    v_{\pi}(s) & = \mathbb{E_{\pi}}[G_t \mid S_t = s] \\
    & = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]
    \end{aligned}
  `}</Latex
>
<p>
  Using the same logic the update step can be adjusted to reflect the recursive
  nature of the value function. The TD update rule uses bootsrapping, meaning
  that the target calculation is also based on an esimation.
</p>
<p>Monte Carlo Update:</p>
<Latex>{String.raw`V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]`}</Latex>
<p>Temporal Difference Update:</p>
<Latex
  >{String.raw`V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]`}</Latex
>
<p>
  Due to recursive notation the target of the update rule, <Latex
    >{String.raw`R_{t+1} + \gamma v_{\pi}(S_{t+1})`}</Latex
  >
  , can be calculated at each single step. That means that it is not necessary to
  wait until the episode finishes to improve the estimate and that temporal difference
  learning is suited for continuing tasks.
</p>

<p class="info">
  The TD-Error quantifies the difference between the bootstrapped target and the
  estimation.
  <br />
  <Latex
    >{String.raw`\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)`}</Latex
  >
</p>

<h4>Algorithm</h4>
<p>
  Compared to Monte Carlo prediction the TD prediction algorithm looks cleaner.
  The interaction step and the update step are consolidated and do not have to
  be put into different loops or functions.
</p>

<Algorithm algoName={"Temporal Difference Prediction"}>
  <AlgorithmState
    >Input: environment <Latex>env</Latex>, policy <Latex
      >{String.raw`\mu`}</Latex
    >
    , state set <Latex>{String.raw`\mathcal{S}`}</Latex>, number of episodes <Latex
      >N</Latex
    >, learning rate <Latex>{String.raw`\alpha`}</Latex>, discount factor <Latex
      >{String.raw`\gamma`}</Latex
    ></AlgorithmState
  >
  <AlgorithmState
    >Initialize: <Latex>V(s)</Latex> for all <Latex
      >{String.raw`s \in \mathcal{S}`}</Latex
    >
    with zeros</AlgorithmState
  >
  <AlgorithmForAll>
    <span slot="condition">episodes <Latex>{String.raw`\in N`}</Latex></span>
    <AlgorithmRepeat>
      <span slot="condition">state is terminal</span>
      <AlgorithmState
        >Generate a new action <Latex>{String.raw`a = \mu(s_t)`}</Latex>
      </AlgorithmState>
      <AlgorithmState
        >Generate a new state and reward <Latex
          >{String.raw`s_{t+1}, r_{t+1}=env(s_t, a_t)`}</Latex
        >
      </AlgorithmState>
      <AlgorithmState>
        <Latex
          >{String.raw`V(S) = V(S) + \alpha [R + \gamma V(S') - V(S)]`}</Latex
        >
      </AlgorithmState>
    </AlgorithmRepeat>
  </AlgorithmForAll>
  <AlgorithmState>Output: value function <Latex>V(s)</Latex></AlgorithmState>
</Algorithm>
<h4>Implementation</h4>
<Code
  code={`
def td_prediction(env, policy, obs_space, num_episodes, alpha, gamma):
    # v as value function
    v = np.zeros(len(obs_space))
    
    for episode in trange(num_episodes):
        # reset variables
        done, obs = False, env.reset()
        
        while not done:
            action = policy(obs)
            next_obs, reward, done, _ = env.step(action)
            v[obs] = v[obs] + alpha * (reward + gamma * v[next_obs] - v[obs])
            obs = next_obs
            
    return v
  `}
/>

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
<Latex
  >{String.raw`Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]`}</Latex
>
<Algorithm algoName={"SARSA"}>
  <AlgorithmState
    >Input: environment <Latex>env</Latex>, state set <Latex
      >{String.raw`\mathcal{S}`}</Latex
    >
    , action set <Latex>{String.raw`\mathcal{A}`}</Latex>, number of episodes <Latex
      >N</Latex
    >
    , learning rate <Latex>{String.raw`\alpha`}</Latex>, discount factor <Latex
      >{String.raw`\gamma`}</Latex
    >
    , epsilon <Latex>{String.raw`\epsilon`}</Latex>
  </AlgorithmState>
  <AlgorithmState
    >Initialize: <Latex>Q(s, a)</Latex> for all <Latex
      >{String.raw`s \in \mathcal{S}`}</Latex
    >
    and
    <Latex>{String.raw`a \in \mathcal{A}`}</Latex>
    with zeros, policy <Latex>{String.raw`\mu(a \mid s)`}</Latex> for all <Latex
      >{String.raw`a \in \mathcal{A}`}</Latex
    >
    where <Latex>{String.raw`\mu`}</Latex> is <Latex
      >{String.raw`\epsilon`}</Latex
    >
    -greedy
  </AlgorithmState>
  <AlgorithmForAll>
    <span slot="condition">episodes <Latex>{String.raw`\in N`}</Latex></span>
    <AlgorithmState
      >Reset state <Latex>S</Latex> and action <Latex>A</Latex>
    </AlgorithmState>
    <AlgorithmRepeat>
      <span slot="condition">state is terminal</span>
      <AlgorithmState
        >Generate tuple <Latex>(R,S',A')</Latex> using policy <Latex
          >{String.raw`\mu`}</Latex
        >
        and MDP <Latex>env</Latex>
      </AlgorithmState>
      <AlgorithmState
        ><Latex
          >{String.raw`Q(S, A) = Q(S) + \alpha [R + \gamma Q(S',A') - Q(S,A)]`}</Latex
        >
      </AlgorithmState>
      <AlgorithmState
        ><Latex>{String.raw`S \leftarrow S', A \leftarrow A'`}</Latex>
      </AlgorithmState>
    </AlgorithmRepeat>
  </AlgorithmForAll>
  <AlgorithmState>Output: value function <Latex>V(s)</Latex></AlgorithmState>
</Algorithm>
<Code
  code={`
def sarsa(env, obs_space, action_space, num_episodes, alpha, gamma, epsilon):
    # q as action value function
    q = np.zeros(shape=(len(obs_space), len(action_space)))
                 
    # epsilon greedy policy
    def policy(obs):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = q[obs].argmax()
        return action
    
    for episode in trange(num_episodes):
        # reset variables
        done, obs = False, env.reset()
        action = policy(obs)
        
        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_action = policy(next_obs)
            
            q[obs][action] = q[obs][action] + alpha * (reward + gamma * q[next_obs][next_action] * (not done) - q[obs][action])
            obs, action = next_obs, next_action
    
    # greedy policy
    policy_mapping = np.argmax(q, axis=1)
    policy = lambda x: policy_mapping[x]
        
    return policy, q
  `}
/>

<h4>Q-Learning</h4>

<p>
  Q-Learning is the On-Policy TD control algorithm. A different policy that is
  used to generate actions is being improved. Theoretically the algorithm should
  be able to learn the optimal control policy from a purely random
  action-selection policy.
</p>
<Latex
  >{String.raw`Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]`}</Latex
>
<Algorithm algoName={"Q-Learning"}>
  <AlgorithmState
    >Input: environment <Latex>env</Latex>, state set <Latex
      >{String.raw`\mathcal{S}`}</Latex
    >
    , action set <Latex>{String.raw`\mathcal{A}`}</Latex>, number of episodes <Latex
      >N</Latex
    >
    , learning rate <Latex>{String.raw`\alpha`}</Latex>, discount factor <Latex
      >{String.raw`\gamma`}</Latex
    >
    , epsilon <Latex>{String.raw`\epsilon`}</Latex>
  </AlgorithmState>
  <AlgorithmState
    >Initialize: <Latex>Q(s,a)</Latex> for all <Latex
      >{String.raw`s \in \mathcal{S}`}</Latex
    >
    and
    <Latex>{String.raw`a \in \mathcal{A}`}</Latex>
    with zeros, policy <Latex>{String.raw`\mu(a \mid s)`}</Latex> for all where <Latex
      >{String.raw`\mu`}</Latex
    > is <Latex>{String.raw`\epsilon`}</Latex>
    -greedy
  </AlgorithmState>
  <AlgorithmForAll>
    <span slot="condition">episodes <Latex>{String.raw`\in N`}</Latex></span>
    <AlgorithmState>Reset state <Latex>S</Latex></AlgorithmState>
    <AlgorithmRepeat>
      <span slot="condition">state is terminal</span>
      <AlgorithmState
        >Generate tuple <Latex>(R,S',A)</Latex> using policy <Latex
          >{String.raw`\mu`}</Latex
        >
        and MDP <Latex>env</Latex>
      </AlgorithmState>
      <AlgorithmState>
        <Latex
          >{String.raw`Q(S, A) = Q(S) + \alpha [R + \gamma \max_aQ(S',a) - Q(S,A)]`}</Latex
        >
      </AlgorithmState>
      <AlgorithmState
        ><Latex>{String.raw`S \leftarrow S'`}</Latex></AlgorithmState
      >
    </AlgorithmRepeat>
  </AlgorithmForAll>
  <AlgorithmState>Output: value function <Latex>V(s)</Latex></AlgorithmState>
</Algorithm>
<Code
  code={`
def q_learning(env, obs_space, action_space, num_episodes, alpha, gamma, epsilon):
    # q as action value function
    q = np.zeros(shape=(len(obs_space), len(action_space)))
                 
    # epsilon greedy policy
    def policy(obs):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = q[obs].argmax()
        return action
    
    for episode in trange(num_episodes):
        # reset variables
        done, obs = False, env.reset()
        
        while not done:
            action = policy(obs)
            next_obs, reward, done, _ = env.step(action)
            next_action = policy(next_obs)
            
            q[obs][action] = q[obs][action] + alpha * (reward + gamma * q[next_obs].max() * (not done) - q[obs][action])
            obs = next_obs
    
    # greedy policy
    policy_mapping = np.argmax(q, axis=1)
    policy = lambda x: policy_mapping[x]
        
    return policy, q
  `}
/>
