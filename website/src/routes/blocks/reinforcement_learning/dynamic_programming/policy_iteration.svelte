<script>
  import Question from "$lib/Question.svelte";
  import Math from "$lib/Math.svelte";
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
  function to measure the performance of a given policy <Math
    latex={String.raw`\pi`}
  /> by estimating the corresponding value function <Math
    latex={String.raw`v_{\pi}`}
  />. Policy improvement on the other hand generates a new policy, that is
  better (or at least equal) than the previous policy. Repeating both steps
  generates optimal policy function <Math latex={String.raw`\pi_*`} /> and value
  function <Math latex={String.raw`v_*`} />.
</p>
<p>Policy Iteration is one such iterative algorithm.</p>
<div class="separator" />

<h2>Policy Evaluation</h2>
<p>
  The goal of policy evaluation is to find the true value function <Math
    latex={String.raw`v_{\pi}`}
  /> of the policy <Math latex={String.raw`\pi`} />.
</p>
<Math
  latex={String.raw`
    \begin{aligned}
    v_{\pi}(s) & \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] \\
    & = \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]
    \end{aligned}
  `}
/>
<p>
  When we have a deterministic policy <Math latex={String.raw`\mu`} /> the expression
  becomes easier to work with.
</p>
<Math
  latex={String.raw`v_{\mu}(s) \doteq \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma v_{\pi}(s')]`}
/>

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
<Math
  latex={String.raw`v_{k+1}(s) \doteq \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma v_{k}(s')]`}
/>
<p>
  At each iteration step <Math latex={String.raw`k+1`} /> the left side of the equation
  is updated by using the state values from the previous iteration and the model
  of the Markov decision process. At this point it should become apparent why the
  Bellman equation is useful. Only the reward from the next time step is required
  to improve the approximation, because all subsequent rewards are already condensed
  into the value function from the next time step. That allows the algorithm to use
  the model to look only one step into the future for the reward and use the approximated
  value function for the next time step. By repeating the update step over and over
  again the rewards are getting embedded into the value function and the approximation
  gets better and better.
</p>
<Algorithm algoName={"Iterative Policy Evaluation"}>
  <AlgorithmState>
    Input: policy <Math latex={String.raw`\mu`} />, model <Math
      latex={String.raw`p`}
    />, state set <Math latex={String.raw`\mathcal{S}`} />, stop criterion <Math
      latex={String.raw`\theta`}
    />
    , discount factor <Math latex={String.raw`\gamma`} />
  </AlgorithmState>
  <AlgorithmState>
    Initialize: <Math latex={`V(s) `} /> and <Math
      latex={String.raw`V_{old}(s)`}
    /> for all <Math latex={String.raw`s \in \mathcal{S}`} /> with zeros
  </AlgorithmState>
  <AlgorithmRepeat>
    <Math slot="condition" latex={String.raw`\Delta < \theta `} />
    <AlgorithmState>
      <Math latex={String.raw`\Delta \leftarrow 0`} />
    </AlgorithmState>
    <Math
      latex={String.raw`V_{old}(s) = V(s)\hspace{0.1cm} for\hspace{0.1cm} all \hspace{0.1cm} s \in \mathcal{S}`}
    />
    <AlgorithmForAll>
      <Math slot="condition" latex={String.raw`s \in \mathcal{S}`} />
      <AlgorithmState>
        <Math
          latex={String.raw`V(s) \leftarrow \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma V_{old}(s')]`}
        />
      </AlgorithmState>
      <AlgorithmState>
        <Math
          latex={String.raw`\Delta \leftarrow \max(\Delta,|V_{old}(s) - V(s)|)`}
        />
      </AlgorithmState>
    </AlgorithmForAll>
  </AlgorithmRepeat>
  <AlgorithmState
    >Output: value function <Math latex={String.raw`V(s)`} /></AlgorithmState
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
  Let us assume that the agent follows a policy <Math
    latex={String.raw`\mu(s)`}
  />. But in the current state <Math latex={String.raw`s`} /> the agent contemplates
  to pick the action <Math latex={String.raw`a`} /> that goes against the policy,
  <Math latex={String.raw`a \neq \mu(s)`} />. After that action the agent will
  stick to the old policy <Math latex={String.raw`\mu(s)`} /> and follow it until
  the terminal state <Math latex={String.raw`T`} />. The value of using the
  action <Math latex={String.raw`a`} /> at state <Math latex={String.raw`s`} /> and
  then following the policy <Math latex={String.raw`\mu(s)`} /> is essentially the
  definition of the action-value function, which plays a key role in the policy improvement
  step.
</p>
<Math
  latex={String.raw`q_{\mu}(s, a) \doteq \mathbb{E}[R_{t+1} + \gamma v_{\mu}(S_{t+1}) \mid S_t = s, A_t = a]`}
/>
<p>
  What if the agent compares <Math latex={String.raw`v_{\mu}(s)`} /> and <Math
    latex={String.raw`q_{\mu}(s, a)`}
  />
  and finds out that taking some action <Math latex={String.raw`a`} /> and then following
  <Math latex={String.raw`\mu`} /> is of higher value than strictly following <Math
    latex={String.raw`\mu`}
  />? Does that imply that the agent should change the policy and always follow
  the state <Math latex={String.raw`a`} /> when facing the state <Math
    latex={String.raw`s`}
  />? It turns out that according to the policy improvement theorem this is
  exactly the case.
</p>
<p>
  In the policy improvement step for at least one of the states we have to find
  an action that would create a higher value. If we find such an action we
  create a new policy
  <Math latex={String.raw`\mu'`} /> that always takes the new action
  <Math latex={String.raw`a`} /> at state <Math latex={String.raw`s`} />. In
  practice for each of the states the agent chooses a so called greedy action,
  which means that the agent chooses the action that maximizes the short term
  gain.
</p>
<Math latex={String.raw`\mu'(s) = \arg\max_a q_{\mu}(s, a)`} />
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
    Input: policy <Math latex={String.raw`\mu`} />, model <Math
      latex={String.raw`p`}
    />, state set <Math latex={String.raw`\mathcal{S}`} />, action set <Math
      latex="{String.raw`\mathcal{A}`},"
    /> stop criterion <Math latex={String.raw`\theta`} />
    , discount factor <Math latex={String.raw`\gamma`} />
  </AlgorithmState>
  <AlgorithmState>
    Initialize: <Math latex={`V(s) `} /> and <Math
      latex={String.raw`V_{old}(s)`}
    /> for all <Math latex={String.raw`s \in \mathcal{S}`} /> with zeros, <Math
      latex={String.raw`\mu(s) \in \mathcal{A}(s)`}
    /> randomly
  </AlgorithmState>
  <AlgorithmRepeat>
    <span slot="condition">policy stable</span>
    <br />
    <!--Policy Evaluation -->
    <AlgorithmState><span>1: Policy Evaluation</span></AlgorithmState>
    <AlgorithmRepeat>
      <Math slot="condition" latex={String.raw`\Delta < \theta `} />
      <AlgorithmState>
        <Math latex={String.raw`\Delta \leftarrow 0`} />
      </AlgorithmState>
      <Math
        latex={String.raw`V_{old}(s) = V(s)\hspace{0.1cm} for\hspace{0.1cm} all \hspace{0.1cm} s \in \mathcal{S}`}
      />
      <AlgorithmForAll>
        <Math slot="condition" latex={String.raw`s \in \mathcal{S}`} />
        <AlgorithmState>
          <Math
            latex={String.raw`V(s) \leftarrow \sum_{s', r}p(s', r \mid s, \mu(s))[r + \gamma V_{old}(s')]`}
          />
        </AlgorithmState>
        <AlgorithmState>
          <Math
            latex={String.raw`\Delta \leftarrow \max(\Delta,|V_{old}(s) - V(s)|)`}
          />
        </AlgorithmState>
      </AlgorithmForAll>
    </AlgorithmRepeat>
    <br />
    <!--Policy Improvement -->
    <AlgorithmState>2: Policy Improvement</AlgorithmState>
    <AlgorithmState
      >policy-stable <Math latex={String.raw`\leftarrow`} /> true</AlgorithmState
    >
    <AlgorithmForAll>
      <Math slot="condition" latex={String.raw`s \in \mathcal{S}`} />
      <AlgorithmState
        ><span>old-action</span>
        <Math latex={String.raw`\leftarrow \mu(s)`} /></AlgorithmState
      >
      <AlgorithmState>
        <Math
          latex={String.raw`\mu(s) \leftarrow \arg\max_a \sum_{s', r}p(s', r \mid s, a)[r + \gamma V(s')]`}
        /></AlgorithmState
      >
      <AlgorithmIf>
        <span slot="condition"
          >old-action <Math latex={String.raw`\neq \mu(s)`} /></span
        >
        <AlgorithmState
          >policy-stable <Math latex={String.raw`\leftarrow`} /> false</AlgorithmState
        >
      </AlgorithmIf>
    </AlgorithmForAll>
    <br />
  </AlgorithmRepeat>
  <AlgorithmState
    >Output: policy function <Math latex={String.raw`\mu(s)`} />, value function <Math
      latex={String.raw`V(s)`}
    /></AlgorithmState
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
