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
  <title>World4AI | Reinforcement Learning | Value Iteration Algorithm</title>
  <meta
    name="description"
    content="Value iteration is an iterative (dynamic programming) algorithm. The algorithm alternates between (truncated) policy evaluation and policy improvement to arrive at the optimal policy and value functions"
  />
</svelte:head>

<h1>Value Iteration</h1>
<Question
  >How can we use value iteration to find optimal value and policy function?</Question
>
<div class="separator" />

<p>
  When we consider policy iteration again, we remember that there are two
  distinct steps, policy evaluation and policy improvement. The policy
  improvement step is a single step, where the new policy is derived by acting
  greedily. The policy evaluation on the other hand is a longer iterative
  process.
</p>

<p>
  It turns out that it is not necessary to wait for the policy evaluation
  algorithm to finish. The value iteration algorithm works with only one step of
  policy evaluation.
</p>

<p>
  Value Iteration is essentially the Bellman optimality equation, transformed
  from equation to an update step.
</p>
<Latex
  >{String.raw`
    \begin{aligned}
    v_{k+1}(s) & \doteq \max_a \mathbb{E}[R_{t+1} + \gamma v_k (S_{t+1}) \mid S_t = s, A_t = a] \\
    & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]
    \end{aligned}
  `}</Latex
>

<p>
  Although the update step looks like a single step it actually combines
  truncated (one step) policy evaluation and policy improvement in a single
  step.
</p>
<Latex
  >{String.raw`q_{k+1}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]`}</Latex
>
<Latex>{String.raw`v_{k+1}(s) = \max_a q_{k+1}(s, a)`}</Latex>

<p>
  In the first step the action-value function is calculated based on the old
  state-value function and the model of the MDP. In the second step a max over
  the action-value function is taken in order to generate the new state-value
  function. That implicitly generates a new policy as a value function is always
  calculated for a particular policy.
</p>
<p>
  The combination of both is the value iteration algorithm. The iterative
  process continues until the difference between the old and the new state-value
  function is smaller than some parameter theta <Latex
    >{String.raw`\theta`}</Latex
  >. As the final step the optimal policy can be deduced using the argmax over
  the optimal action-value function.
</p>
<Algorithm algoName={"Value Iteration"}>
  <AlgorithmState>
    Input: model <Latex>p</Latex>, state set <Latex
      >{String.raw`\mathcal{S}`}</Latex
    >
    , action set <Latex>{String.raw`\mathcal{A}`}</Latex> stop criterion <Latex
      >{String.raw`\theta`}</Latex
    >
    , discount factor <Latex>{String.raw`\gamma`}</Latex>
  </AlgorithmState>

  <AlgorithmState>
    Initialize: <Latex>V(s)</Latex> and <Latex>{String.raw`V_{old}(s)`}</Latex>,
    for all <Latex>{String.raw`s \in \mathcal{S}`}</Latex> with zeros
  </AlgorithmState>
  <AlgorithmRepeat>
    <Latex slot="condition">{String.raw`\Delta < \theta`}</Latex>
    <AlgorithmState>
      <Latex>{String.raw`\Delta\leftarrow 0`}</Latex></AlgorithmState
    >
    <AlgorithmState
      ><Latex>{String.raw`V_{old}(s) = V(s)`}</Latex>
    </AlgorithmState>
    <AlgorithmForAll>
      <Latex slot="condition">{String.raw`s \in \mathcal{S}`}</Latex>
      <AlgorithmForAll>
        <Latex slot="condition">{String.raw`a \in \mathcal{A}`}</Latex>
        <AlgorithmState
          ><Latex
            >{String.raw`Q(s, a) \leftarrow \sum_{s', r}p(s', r \mid s, a)[r + \gamma V_{old}(s')]`}</Latex
          >
        </AlgorithmState>
      </AlgorithmForAll>
      <AlgorithmState>
        <Latex>{String.raw`V(s) \leftarrow \max_a Q(s, a)`}</Latex>
      </AlgorithmState>
      <AlgorithmState>
        <Latex
          >{String.raw`\Delta \leftarrow \max(\Delta,|V_{old}(s) - V(s)|)`}</Latex
        >
      </AlgorithmState>
    </AlgorithmForAll>
  </AlgorithmRepeat>
  <AlgorithmState>
    <Latex>{String.raw`\mu(s) = \arg\max_a Q(s, a)`}</Latex>
  </AlgorithmState>
  <AlgorithmState>
    Output: value function
    <Latex>V(s)</Latex>
    and policy
    <Latex>{String.raw`\mu(s)`}</Latex>
  </AlgorithmState>
</Algorithm>
<Code
  code={`def value_iteration(obs_space, action_space, model, theta, gamma):
    # initialize value function with zeros
    value_function = [0 for _ in obs_space]
    policy = {}

    while True:
        max_delta = 0
        value_function_old = value_function.copy()
        action_value_function = [[0 for action in action_space] for obs in obs_space]
        for obs in obs_space:
            # the below two variables are requred for the policy
            argmax = 0
            v_max = 0
            for action in action_space:
                for prob, next_obs, reward, done in model[obs][action]:
                    action_value_function[obs][action]+=prob*(reward + gamma*value_function_old[next_obs] * (not done))
                
                if action_value_function[obs][action] > v_max:
                    v_max = action_value_function[obs][action]
                    argmax = action
            
            policy[obs] = argmax
            value_function[obs] = max(action_value_function[obs])
            
            delta = abs(value_function[obs] - value_function_old[obs])
            if delta > max_delta:
                max_delta = delta
    
        # break condition
        if max_delta < theta:
            return value_function, policy`}
/>
<div class="separator" />
