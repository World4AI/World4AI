<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Value Iteration Algorithm</title>
  <meta
    name="description"
    content="Value iteration is an iterative (dynamic programming) algorithm. The algorithm alternates between (truncated) policy evaluation and policy improvement to arrive at the optimal policy and value functions"
  />
</svelte:head>

<h1>Value Iteration</h1>
<div class="separator" />

<Container>
  <p>
    When we consider policy iteration again, we should remember that there are
    two distinct steps, policy evaluation and policy improvement. The policy
    improvement step is a single step, where the new policy is derived by acting
    greedily. The policy evaluation on the other hand is a longer iterative
    process. It turns out that it is not necessary to wait for the policy
    evaluation algorithm to finish converging to the true value function. In
    fact the value iteration algorithm works with only one single policy
    evaluation step.
  </p>
  <p>
    The main goal of value iteration is to find the optimal value function <Latex
      >v_*(s)</Latex
    >, that can be used to derive the optimal policy. The optimal value function
    can be expressed as a Bellman equation that looks as follows.
  </p>
  <Latex>
    {String.raw`
\begin{aligned}
  v_*(s) & = \max_a q_*(s, a) \\
  & = \max_a \mathbb{E}_{\pi}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] \\ 
  & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_*(s')]
\end{aligned}
`}
  </Latex>
  <p>
    The value iteration is essentially the Bellman optimality equation, that has
    been transformed to an iterative algorithm.
  </p>
  <Latex
    >{String.raw`
      v_{k+1}(s) \doteq \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]
  `}</Latex
  >
  <p>
    Although the update step looks like a single step at first glance, it
    actually combines truncated (one step) policy evaluation and policy
    improvement in a single step.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
    \text{(1: Policy Evaluation) } & q_{k+1}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')] \\
    \text{(2: Policy Improvement) }& v_{k+1}(s) = \max_a q_{k+1}(s, a)
  \end{aligned}
`}
  </Latex>

  <p>
    In the first step the action-value function is calculated based on the old
    state-value function and the model of the Markov decision process. In the
    second step a max over the action-value function is taken in order to
    generate the new state-value function. That implicitly generates a new
    policy as a value function is always calculated for a particular policy.
  </p>
  <p>
    The combination of both steps is the value iteration algorithm. The
    iterative process continues until the difference between the old and the new
    state-value function is smaller than some parameter theta <Latex
      >{String.raw`\theta`}</Latex
    >. As the final step the optimal policy can be deduced by always selecting
    the greedy action.
  </p>
  <p>
    Below is the Python implementation of the value iteration algorithm.
    Compared to policy iteration the implementation is more compact, because
    policy evaluation and policy improvement can be implemented in a single
    function.
  </p>
  <div class="separator" />
</Container>
