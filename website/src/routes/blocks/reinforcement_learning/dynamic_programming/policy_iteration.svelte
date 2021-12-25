<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Code from "$lib/Code.svelte";
  import Algorithm from "$lib/algorithm/Algorithm.svelte";
  import AlgorithmState from "$lib/algorithm/AlgorithmState.svelte";
  import AlgorithmRepeat from "$lib/algorithm/AlgorithmRepeat.svelte";
  import AlgorithmForAll from "$lib/algorithm/AlgorithmForAll.svelte";
  import AlgorithmIf from "$lib/algorithm/AlgorithmIf.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Policy Iteration Algorithm</title>
  <meta
    name="description"
    content="Policy iteration is an iterative (dynamic programming) algorithm. The algorithm alternates between policy evaluation and policy improvement to arrive at the optimal policy and value functions"
  />
</svelte:head>

<h1>Policy Iteration</h1>
<Question
  >How can we use policy iteration to find optimal policy and value functions?</Question
>
<div class="separator" />
<p>
  In reinforcement learning iterative algorithms usually consist of two basic
  steps: policy evaluation and policy improvement. Policy evaluation has the
  function to measure the performance of a given policy <Latex
    >{String.raw`\pi`}</Latex
  >
  by estimating the corresponding value function <Latex>{String.raw`\pi`}</Latex
  >
  . Policy improvement on the other hand generates a new policy, that is better (or
  at least equal) than the previous policy. Repeating both steps generates optimal
  policy function <Latex>{String.raw`\pi_*`}</Latex> and value function <Latex
    >{String.raw`v_*`}</Latex
  >.
</p>
<p>Policy Iteration is one such iterative algorithm.</p>
<div class="separator" />

<h2>Policy Evaluation</h2>
<p>
  The goal of policy evaluation is to find the true value function <Latex
    >{String.raw`v_{\pi}`}</Latex
  >
  of the policy <Latex>{String.raw`\pi`}</Latex>.
</p>
<Latex
  >{String.raw`
    \begin{aligned}
    v_{\pi}(s) & \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] \\
    & = \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]
    \end{aligned}
  `}</Latex
>
<p>
  When we have a deterministic policy <Latex>{String.raw`\mu`}</Latex> the expression
  becomes easier to work with.
</p>
<Latex
  >{String.raw`v_{\mu}(s) \doteq \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma v_{\pi}(s')]`}</Latex
>
<p>
  When we start the policy evaluation algorithm, the first step is to generate a
  value function that can be used to improve. The initial values are set either
  randomly or to zero. When we start to use the above equation we will not
  surprisingly discover that the random/zero value (the left side of the above
  equation) and the expected value of the reward plus value for the next state
  (the right side of the above equation) will diverge quite a lot. The goal of
  the policy evaluation algorithm is to make the left side of the equation and
  the right side of the equation be exactly equal. That is done in an iterative
  process where at each step the difference between both sides is reduced. In
  practice we do not expect the difference between the two to go all the way
  down to zero. Instead we define a threshold value. For example a threshold
  value of 0.0001 indicates that we can interrupt the iterative process as soon
  as for all of the states the difference between the left and the right side of
  the equation is below the value.
</p>
<Latex
  >{String.raw`v_{k+1}(s) \doteq \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma v_{k}(s')]`}</Latex
>
<p>
  At each iteration step <Latex>k+1</Latex> the left side of the equation is updated
  by using the state values from the previous iteration and the model of the Markov
  decision process. At this point it should become apparent why the Bellman equation
  is useful. Only the reward from the next time step is required to improve the approximation,
  because all subsequent rewards are already condensed into the value function from
  the next time step. That allows the algorithm to use the model to look only one
  step into the future for the reward and use the approximated value function for
  the next time step. By repeating the update step over and over again the rewards
  are getting embedded into the value function and the approximation gets better
  and better.
</p>
<Algorithm algoName={"Iterative Policy Evaluation"}>
  <AlgorithmState>
    Input: policy <Latex>{String.raw`\mu`}</Latex>, model <Latex
      >{String.raw`p`}</Latex
    >
    , state set <Latex>{String.raw`\mathcal{S}`}</Latex>, stop criterion <Latex
      >{String.raw`\theta`}</Latex
    >
    , discount factor <Latex>{String.raw`\gamma`}</Latex>
  </AlgorithmState>
  <AlgorithmState>
    Initialize: <Latex>V(s)</Latex> and <Latex>{String.raw`V_{old}(s)`}</Latex> for
    all <Latex>{String.raw`s \in \mathcal{S}`}</Latex> with zeros
  </AlgorithmState>
  <AlgorithmRepeat>
    <Latex slot="condition">{String.raw`\Delta < \theta`}</Latex>
    <AlgorithmState>
      <Latex>{String.raw`\Delta \leftarrow 0`}</Latex>
    </AlgorithmState>
    <Latex
      >{String.raw`V_{old}(s) = V(s)\hspace{0.1cm} for\hspace{0.1cm} all \hspace{0.1cm} s \in \mathcal{S}`}</Latex
    >
    <AlgorithmForAll>
      <Latex slot="condition">{String.raw`s \in \mathcal{S}`}</Latex>
      <AlgorithmState>
        <Latex
          >{String.raw`V(s) \leftarrow \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma V_{old}(s')]`}</Latex
        >
      </AlgorithmState>
      <AlgorithmState>
        <Latex
          >{String.raw`\Delta \leftarrow \max(\Delta,|V_{old}(s) - V(s)|)`}</Latex
        >
      </AlgorithmState>
    </AlgorithmForAll>
  </AlgorithmRepeat>
  <AlgorithmState
    >Output: value function <Latex>{String.raw`V(s)`}</Latex></AlgorithmState
  >
</Algorithm>
<Code
  code={`def policy_evaluation(obs_space, model, policy, theta, gamma):
    # initialize value function with zeros
    value_function = [0 for _ in obs_space]
    
    while True:
        max_delta = 0
        value_function_old = value_function.copy()
        for obs in obs_space:
            action = policy[obs]
            v = 0
            for prob, next_obs, reward, done in model[obs][action]:
                v+=prob*(reward + gamma*value_function_old[next_obs] * (not done))
            value_function[obs] = v
            
            delta = abs(v - value_function_old[obs])
            if delta > max_delta:
                max_delta = delta
        
        # break condition
        if max_delta < theta:
            break
    
    return value_function`}
/>
<div class="separator" />

<h2>Policy Improvement</h2>
<p>
  Let us assume that the agent follows a policy <Latex
    >{String.raw`\mu(s)`}</Latex
  >. But in the current state <Latex>s</Latex> the agent contemplates to pick the
  action <Latex>a</Latex> that goes against the policy,
  <Latex>{String.raw`a \neq \mu(s)`}</Latex>. After that action the agent will
  stick to the old policy <Latex>{String.raw`\mu(s)`}</Latex> and follow it until
  the terminal state <Latex>T</Latex>. The value of using the action <Latex
    >a</Latex
  > at state <Latex>s</Latex> and then following the policy <Latex
    >{String.raw`\mu(s)`}</Latex
  > is essentially the definition of the action-value function, which plays a key
  role in the policy improvement step.
</p>
<Latex
  >{String.raw`q_{\mu}(s, a) \doteq \mathbb{E}[R_{t+1} + \gamma v_{\mu}(S_{t+1}) \mid S_t = s, A_t = a]`}</Latex
>
<p>
  What if the agent compares <Latex>{String.raw`v_{\mu}(s)`}</Latex> and <Latex
    >{String.raw`q_{\mu}(s, a)`}</Latex
  >
  and finds out that taking some action <Latex>a</Latex> and then following
  <Latex>\mu</Latex> is of higher value than strictly following <Latex
    >{String.raw`\mu`}</Latex
  >
  ? Does that imply that the agent should change the policy and always follow the
  state <Latex>a</Latex> when facing the state <Latex>s</Latex>? It turns out
  that according to the policy improvement theorem this is exactly the case.
</p>
<p>
  In the policy improvement step for at least one of the states we have to find
  an action that would create a higher value. If we find such an action we
  create a new policy
  <Latex>{String.raw`\mu'`}</Latex> that always takes the new action
  <Latex>a</Latex> at state <Latex>s</Latex>. In practice for each of the states
  the agent chooses a so called greedy action, which means that the agent
  chooses the action that maximizes the short term gain.
</p>
<Latex>{String.raw`\mu'(s) = \arg\max_a q_{\mu}(s, a)`}</Latex>
<Code
  code={`def policy_improvement(obs_space, action_space, model, value_function, policy, gamma):
    new_policy = policy.copy()
    for obs in obs_space:
        v_max = 0
        argmax = 0
        
        for action in action_space:
            v = 0
            for prob, next_obs, reward, done in model[obs][action]:
                v+=prob*(reward + gamma*value_function[next_obs] * (not done))
            if v > v_max:
                v_max = v
                argmax = action
        new_policy[obs] = argmax
    return new_policy`}
/>
<div class="separator" />

<h2>The Policy Iteration Algorithm</h2>
<p>
  The idea of policy iteration is to alternate between policy evaluation and
  policy improvement until the optimal policy has been reached. Once the new
  policy and the old policy are exactly the same we have reached the optimal
  policy.
</p>
<Algorithm algoName={"Policy Iteration"}>
  <AlgorithmState>
    Input: policy <Latex>{String.raw`\mu`}</Latex>, model <Latex>p</Latex>,
    state set <Latex>{String.raw`\mathcal{S}`}</Latex>, action set <Latex
      >{String.raw`\mathcal{A}`}</Latex
    >
    stop criterion <Latex>{String.raw`\theta`}</Latex>, discount factor <Latex
      >{String.raw`\gamma`}</Latex
    >
  </AlgorithmState>
  <AlgorithmState>
    Initialize: <Latex>V(s)</Latex> and <Latex>{String.raw`V_{old}(s)`}</Latex>
    for all <Latex>{String.raw`s \in \mathcal{S}`}</Latex> with zeros, <Latex
      >{String.raw`\mu(s) \in \mathcal{A}(s)`}</Latex
    > randomly
  </AlgorithmState>
  <AlgorithmRepeat>
    <span slot="condition">policy stable</span>
    <br />
    <!--Policy Evaluation -->
    <AlgorithmState><span>1: Policy Evaluation</span></AlgorithmState>
    <AlgorithmRepeat>
      <Latex slot="condition">{String.raw`\Delta < \theta `}</Latex>
      <AlgorithmState>
        <Latex>{String.raw`\Delta \leftarrow 0`}</Latex>
      </AlgorithmState>
      <Latex
        >{String.raw`V_{old}(s) = V(s)\hspace{0.1cm} for\hspace{0.1cm} all \hspace{0.1cm} s \in \mathcal{S}`}</Latex
      >
      <AlgorithmForAll>
        <Latex slot="condition">{String.raw`s \in \mathcal{S}`}</Latex>
        <AlgorithmState>
          <Latex
            >{String.raw`V(s) \leftarrow \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma V_{old}(s')]`}</Latex
          >
        </AlgorithmState>
        <AlgorithmState>
          <Latex
            >{String.raw`\Delta \leftarrow \max(\Delta,|V_{old}(s) - V(s)|)`}</Latex
          >
        </AlgorithmState>
      </AlgorithmForAll>
    </AlgorithmRepeat>
    <br />
    <!--Policy Improvement -->
    <AlgorithmState>2: Policy Improvement</AlgorithmState>
    <AlgorithmState
      >policy-stable <Latex>{String.raw`\leftarrow`}</Latex> true</AlgorithmState
    >
    <AlgorithmForAll>
      <Latex slot="condition">{String.raw`s \in \mathcal{S}`}</Latex>
      <AlgorithmState
        ><span>old-action</span>
        <Latex>{String.raw`\leftarrow \mu(s)`}</Latex>
      </AlgorithmState>
      <Latex
        >{String.raw`\mu(s) \leftarrow \arg\max_a \sum_{s', r}p(s', r \mid s, a)[r + \gamma V(s')]`}</Latex
      >
      <AlgorithmIf>
        <span slot="condition"
          >old-action <Latex>{String.raw`\neq \mu(s)`}</Latex></span
        >
        <AlgorithmState
          >policy-stable <Latex>{String.raw`\leftarrow`}</Latex> false</AlgorithmState
        >
      </AlgorithmIf>
    </AlgorithmForAll>
    <br />
  </AlgorithmRepeat>
  <AlgorithmState
    >Output: policy function <Latex>{String.raw`\mu(s)`}</Latex>, value function <Latex
      >V(s)</Latex
    ></AlgorithmState
  >
</Algorithm>

<Code
  code={`def policy_iteration(obs_space, action_space, model, policy, theta, gamma):
    while True:
        value_function = policy_evaluation(obs_space, model, policy, theta, gamma)
        new_policy = policy_improvement(obs_space, action_space, model, value_function, policy, gamma)
        
        if policy==new_policy:
            return value_function, policy
        
        policy = new_policy`}
/>
<div class="separator" />
<h2>Tutorial</h2>
<p>
  The full implementation of the policy iteration algorithm can be found in our
  official <a
    target="_blank"
    href="https://github.com/World4AI/World4AI/tree/main/tutorials"
    >github repo</a
  >.
</p>
<div class="separator" />
