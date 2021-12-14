<script>
  import Question from "$lib/Question.svelte";
  import Math from "$lib/Math.svelte";
  import Code from "$lib/Code.svelte";
  import Algorithm from "$lib/algorithm/Algorithm.svelte";
  import AlgorithmState from "$lib/algorithm/AlgorithmState.svelte";
  import AlgorithmRepeat from "$lib/algorithm/AlgorithmRepeat.svelte";
  import AlgorithmForAll from "$lib/algorithm/AlgorithmForAll.svelte";
</script>

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

<Math
  latex={String.raw`v_{\pi}(s) \doteq \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]`}
/>

<Math
  latex={String.raw`v_{k+1}(s) \doteq \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{k}(s')]`}
/>
<Algorithm algoName={"Iterative Policy Iteration"}>
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
<div class="separator" />
