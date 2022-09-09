<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";
  import Button from "$lib/Button.svelte";
  import { PolicyIteration } from "$lib/reinforcement_learning/grid_world/PolicyIteration";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import { gridMap } from "$lib/reinforcement_learning/grid_world/maps";

  let env = new GridEnvironment(gridMap, true);
  let policyIteration = new PolicyIteration(
    env.observationSpace,
    env.actionSpace,
    env.getModel(),
    0.00001,
    0.99
  );

  const cellsStore = env.cellsStore;
  $: cells = $cellsStore;

  const valueStore = policyIteration.valueStore;
  const policyStore = policyIteration.policyStore;
  $: valueFunction = $valueStore;
  $: policy = $policyStore;

  let showPolicy = true;
  function switchDisplay() {
    showPolicy = !showPolicy;
  }

  function policyEvaluationStep() {
    policyIteration.policyEvaluationStep();
  }

  function policyEvaluation() {
    policyIteration.policyEvaluation();
  }

  function executePolicyIteration() {
    policyIteration.policyIteration();
  }
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Policy Iteration Algorithm</title>
  <meta
    name="description"
    content="Policy iteration is an iterative (dynamic programming) algorithm. The algorithm alternates between policy evaluation and policy improvement to arrive at the optimal policy and value functions"
  />
</svelte:head>

<h1>Policy Iteration</h1>
<div class="separator" />

<Container>
  <p>
    Dynamic programming algorithms that are designed to solve a Markov decision
    process are iterative algorithms, which consist of two basic steps: policy
    evaluation and policy improvement. The purpose of policy evaluation is to
    measure the performance of a given policy <Latex>{String.raw`\pi`}</Latex>
    by estimating the corresponding value function <Latex
      >{String.raw`v_{\pi}(s)`}</Latex
    >
    . Policy improvement on the other hand generates a new policy, that is better
    (or at least not worse) than the previous policy. The output of policy evaluation
    is used as an input into policy improvement and vice versa. The iterative process
    of evaluation and improvement produces value and policy functions that converge
    towards the optimal policy function <Latex>{String.raw`\pi_*`}</Latex> and optimal
    value function <Latex>{String.raw`v_*`}</Latex> over time. The policy iteration
    algorithm that is covered in this section is one such iterative algorithm.
  </p>
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
   v_{\pi}(s)  & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(s') \mid S_t=s] \\
& = \sum_a \pi(a \mid s)  R(a, s) + \gamma \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)v_{\pi}(s')
\end{aligned}
`}</Latex
  >
  <p>
    Often it is more convenient to use the joint probability of simultaneously
    getting the reward <Latex>r</Latex> and the next state <Latex>s'</Latex> given
    current state <Latex>s</Latex> and action <Latex>a</Latex>. This joint
    probability function is depicted as <Latex>p(s', r \mid s, a)</Latex>. This
    notation is more compact and is likely to make the transition from theory to
    practice easier.
  </p>
  <Latex
    >{String.raw`
    \begin{aligned}
    v_{\pi}(s) & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] \\
    & = \sum_a \pi(a \mid s)  R(a, s) + \gamma \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)v_{\pi}(s') \\
    & = \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]
    \end{aligned}
  `}</Latex
  >
  <p>
    If we look closely at the Bellman equation, we can observe that the equation
    basically consists of two sides. The left side and the right side.
  </p>
  <Latex
    >{String.raw`\underbrace{v_{\pi}(s)}_{\text{left side}} = \underbrace{\sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]}_{\text{right side}}`}</Latex
  >
  <p>
    The left side is the function that returns the value of a state <Latex
      >s</Latex
    >, it is the mapping from states to values. The left side is essentially the
    function we are trying to find. The right side is the definition of the
    value function, that is based on the expectation of the returns and is
    expressed using the Bellman equation.
  </p>
  <p>
    When we initialize the policy evaluation algorithm, the first step is to
    generate a value function that is used as a benchmark that needs to be
    constantly improved in the interative process. The initial values are set
    either randomly or to zero. When we start to use the above equation we will
    not surprisingly discover that the random/zero value (the left side of the
    above equation) and the expected value of the reward plus value for the next
    state (the right side of the above equation) will diverge quite a lot. The
    goal of the policy evaluation algorithm is to make the left side of the
    equation and the right side of the equation to be exactly equal. That is
    done in an iterative process where at each step the difference between both
    sides is reduced. In practice we do not expect the difference between the
    two to go all the way down to zero. Instead we define a threshold value. For
    example a threshold value of 0.0001 indicates that we can interrupt the
    iterative process as soon as for all the states <Latex
      >{String.raw`s \in \mathcal{S}`}</Latex
    > the difference between the left and the right side of the equation is below
    the threshold value.
  </p>
  <p>
    The policy estimation algorithm is relatively straightforward. All we need
    to do is to turn the definition of the of the Bellman equation into the
    update rule.
  </p>
  <Latex
    >{String.raw`V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s', r}p(s', r \mid s, a)[r + \gamma V_{k}(s')]`}</Latex
  >
  <p>
    Above we use <Latex>V(s)</Latex> instead of
    <Latex>v(s)</Latex>. This notational difference is to show that <Latex
      >v(s)</Latex
    > is the true value function of a policy <Latex>\pi</Latex>, while <Latex
      >V(s)</Latex
    > is it's estimate. At each iteration step <Latex>k+1</Latex> the left side of
    the equation (the esimate of the value function) is replaced by the right hand
    of the equation. At this point it should become apparent why the Bellman equation
    is useful. Only the reward from the next time step is required to improve the
    approximation, because all subsequent rewards are already condensed into the
    value function from the next time step. That allows the algorithm to use the
    model to look only one step into the future for the reward and use the approximated
    value function for the next time step. By repeating the update rule over and
    over again the rewards are getting embedded into the value function and the approximation
    gets better and better.
  </p>
  <p class="info">
    The process of using past estimates to improve current estimates is called
    <strong>bootstrapping</strong>. Bootstrapping is used heavily through
    reinforcement learning and can generally be used without the full knowledge
    of the model of the environment.
  </p>
  <p>
    Below you can find the Python implementation of the policy evaluation
    algorithm.
  </p>
  <p>
    Once again below we deal with a simple grid world where the task is to
    arrive at the bottom left corner starting from the top left corner. The
    environment transitions with 50% probability into the desired direction
    (unless there is some barrier) and with 50% chance the environment takes a
    randomm action. The playground below allows you to calculate the value
    function for a randomly initialized deterministic policy, using the policy
    evaluation algorithm. You can switch between the display of the policy and
    the value function. Start by taking one step of the algorithm at the time
    and observe how the value function propagates and the difference between
    steps keeps decreasing. Finally you can run the full policy evaluation
    algorithm, where the iterative process keeps going until the difference
    betwenn the left and the right side of the Bellman equation is less than
    0.00001.
  </p>
  <div class="flex-space">
    <Grid
      {cells}
      valueFunction={!showPolicy ? valueFunction : null}
      policy={showPolicy ? policy : null}
    />
    <div class="flex-vertical">
      <Button
        value={showPolicy ? "Swith To Value Function" : "Swith To Policy"}
        on:click={switchDisplay}
      />
      <Button
        value={"1 Step Policy Evaluation"}
        on:click={policyEvaluationStep}
      />
      <Button value={"Full Policy Evaluation"} on:click={policyEvaluation} />
    </div>
  </div>

  <div class="separator" />

  <h2>Policy Improvement</h2>
  <p>
    The goal of policy improvement is to create a new and improved policy using
    the value function <Latex>{String.raw`V(s)`}</Latex> from the previous policy
    evaluation step.
  </p>
  <p>
    Let us assume for simplicity that the agent follows a deterministic policy <Latex
      >{String.raw`\mu(s)`}</Latex
    >, but in the current state <Latex>s</Latex> the agent contemplates to pick the
    action <Latex>a</Latex> that contradicts the policy, therefore
    <Latex>{String.raw`a \neq \mu(s)`}</Latex>. After that action the agent will
    stick to the old policy <Latex>{String.raw`\mu(s)`}</Latex> and follow it until
    the terminal state <Latex>T</Latex>. We can measure the value of using the
    action <Latex>a</Latex> at state <Latex>s</Latex> and then following the policy
    <Latex>{String.raw`\mu(s)`}</Latex> using the action-value function.
  </p>
  <Latex
    >{String.raw`q_{\mu}(s, a) \doteq \mathbb{E}[R_{t+1} + \gamma v_{\mu}(S_{t+1}) \mid S_t = s, A_t = a]`}</Latex
  >
  <p>
    What if the agent compares the estimates <Latex
      >{String.raw`V_{\mu}(s)`}</Latex
    > and <Latex>{String.raw`Q_{\mu}(s, a)`}</Latex>
    and determines that taking some action <Latex>a</Latex> and then following
    <Latex>\mu</Latex> is of higher value than strictly following <Latex
      >{String.raw`\mu`}</Latex
    >, showing that
    <Latex>{String.raw`Q_{\mu}(s, a) > V_{\mu}(s)`}</Latex>? Does that imply
    that the agent should change the policy and always take the action <Latex
      >a</Latex
    >
    when facing the state <Latex>s</Latex>? Does the short term gain from the
    new action <Latex>a</Latex> justifies changing the policy? It turns out that
    this is exactly the case.
  </p>
  <p>
    In the policy improvement step the we create a new policy <Latex
      >{String.raw`\mu'`}</Latex
    > where the agent chooses the greedy action at each state <Latex
      >{String.raw`s \in \mathcal{S}`}</Latex
    >.
  </p>
  <Latex>{String.raw`\mu'(s) = \arg\max_a Q_{\mu}(s, a)`}</Latex>
  <p>Below you can find a Python example of the policy improvement step.</p>
  <div class="separator" />

  <h2>The Policy Iteration Algorithm</h2>
  <p>
    The idea of policy iteration is to alternate between policy evaluation and
    policy improvement until the optimal policy has been reached. Once the new
    policy and the old policy are exactly the same we have reached the optimal
    policy.
  </p>

  <p>
    Below is a playground from the same gridworld, that demonstrates the policy
    iteration algorithm. The algorithm finds the optimal policy and
    corresponding optimal value function, once you click on the <em
      >"policy iteration"</em
    > button.
  </p>
  <div class="flex-space">
    <Grid
      {cells}
      valueFunction={!showPolicy ? valueFunction : null}
      policy={showPolicy ? policy : null}
    />
    <div class="flex-vertical">
      <Button
        value={showPolicy ? "Swith To Value Function" : "Swith To Policy"}
        on:click={switchDisplay}
      />
      <Button value={"Policy Iteration"} on:click={executePolicyIteration} />
    </div>
  </div>
  <div class="separator" />
</Container>
