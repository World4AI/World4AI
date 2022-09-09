<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Trajectory from "$lib/Trajectory.svelte";
  import Discounting from "../_solution/Discounting.svelte";
  let trajectory = [
    {
      type: "R",
      subscript: 1,
    },
    {
      type: "R",
      subscript: 2,
    },
    {
      type: "R",
      subscript: 3,
    },
    {
      type: "R",
      subscript: 4,
    },
    {
      type: "R",
      subscript: 5,
    },
    {
      type: "R",
      subscript: 6,
    },
    {
      type: "R",
      subscript: 7,
    },
    {
      type: "R",
      subscript: "8",
    },
  ];
  let trajectoryNumbers = [
    {
      type: "-1",
      subscript: 1,
    },
    {
      type: "-1",
      subscript: 2,
    },
    {
      type: "-1",
      subscript: 3,
    },
    {
      type: "-1",
      subscript: 4,
    },
    {
      type: "-1",
      subscript: 5,
    },
    {
      type: "-1",
      subscript: 6,
    },
    {
      type: "-1",
      subscript: 7,
    },
    {
      type: "10",
      subscript: "8",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | MDP Solution</title>
  <meta
    name="description"
    content="A Markov decision process is considered to be solved once the agent found the optimal policy and the optimal value function."
  />
</svelte:head>

<h1>MDP Solution</h1>
<div class="separator" />

<Container>
  <p>
    Once a particular Markov decision process has been defined the next logical
    step is to solve that MDP. In this chapter we are going to discuss <strong
      >what</strong
    >
    it actually means to solve the MDP, while the coming chapters are going to focus
    on <strong>how</strong> and <strong>if</strong> we can find a solution.
  </p>
  <div class="separator" />

  <h2>Return</h2>
  <p>
    A return at timestep <Latex>t</Latex>, denoted as <Latex>G_t</Latex>, is
    simply the sum of rewards starting from timestep <Latex>t+1</Latex> and going
    either to terminal state <Latex>T</Latex> (episodic tasks) or to infinity (continuing
    tasks). The letter <Latex>G</Latex> stands for
    <em>Goal</em>, because in reinforcement learning the goal of the environment
    is encoded in the rewards.
  </p>
  <p>
    In episodic tasks the return is the sum of rewards in a single episode from
    time step <Latex>t+1</Latex> to the terminal time step <Latex>T</Latex>.
  </p>
  <Latex>{String.raw`G_t = R_{t+1} + R_{t+2} + … + R_T`}</Latex>
  <p>
    In continuing tasks the return is the sum of rewards starting at time step <Latex
      >t+1</Latex
    >
    and going to possibly infinity.
  </p>
  <Latex
    >{String.raw`G_t = R_{t+1} + R_{t+2} + R_{t+3} + …  = \sum_{k=0}^\infty{R_{k+t+1}}`}</Latex
  >
  <p>
    In the exemplified episodic task below the Markov decision process plays
    through the sequence of states, actions and rewards all the way to the
    terminal state <Latex>T</Latex>. This sequence of states, actions and
    rewards of an episode is called a <em>trajectory</em>.
  </p>
  <Trajectory />
  <p>
    The episode goes from timestep
    <Latex>0</Latex> until the terminal timestep <Latex>T = 8</Latex>. For each
    timestep taken, the agent receives a negative reward of -1 and a reward of
    +10 when the agent reaches the terminal state <Latex>S_T</Latex>.
  </p>
  <Trajectory {trajectory} />
  <Trajectory trajectory={trajectoryNumbers} />
  <p>
    Based on that trajectory the returns from the perspective of time step 0, 1,
    and 2 look as follows.
  </p>
  <Latex
    >{String.raw`
   G_0  = R_1 + R_2 + R_3 + R_4 + R_5 + R_6 + R_7 + R_8 \\
   = (-1) + (-1) + (-1) + (-1) + (-1) + (-1) + (-1) + 10 = 3
`}</Latex
  >
  <Latex
    >{String.raw`
   G_1 = R_2 + R_3 + R_4 + R_5 + R_6 + R_7 + R_8 \\
   = (-1) + (-1) + (-1) + (-1) + (-1) + (-1) + 10 = 4
`}</Latex
  >
  <Latex
    >{String.raw`
   G_2 = R_3 + R_4 + R_5 + R_6 + R_7 + R_8 \\
   = (-1) + (-1) + (-1) + (-1) + (-1) + 10 = 5
`}</Latex
  >
  <div class="separator" />

  <h2>Discounted Return</h2>
  <p>
    Gamma <Latex>{String.raw`\gamma`}</Latex>
    (also called discount factor) is a value between 0 and 1 (usually between 0.9
    and 0.99) that is used to reduce the value of future rewards, because future
    rewards are considered of less value to the agent than present rewards.
  </p>
  <p class="info">
    <Latex>{String.raw`\gamma`}</Latex> (gamma) is the discount factor, where <Latex
      >{String.raw`0 \leq \gamma \leq 1`}</Latex
    >.
  </p>
  <p>
    Let us consider the following example to convince ourselves that future
    rewards are indeed of less value than present rewards. If we have to choose
    between getting 1000$ now and getting 1000$ in 10 years, we should
    definetely choose the 1000$ in the present. The money could be invested in a
    10 year risk free bond, so that at the end of the 10 year period the
    investor gets back an amout that is larger than 1000$. In reinforcement
    learning this concept called <em>"time value of money"</em> is extended to all
    other types of rewards. Specificylly in reinforcement learning we use the geometric
    series to calculate the current value of a future reward.
  </p>
  <Latex>S = 1 + \gamma + \gamma^2 + \gamma^3 + \gamma^4 + ...</Latex>
  <p>Therefore the discounted return looks as follows:</p>
  <Latex
    >{String.raw`G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...`}</Latex
  >
  <p>
    Mathematically the discounting is necessary in continuing tasks. If for
    example at each timestep the agent receives a reward of +1 and the episode
    has no natural ending, the return <Latex>G</Latex> could theoretically become
    infinite. Due to the geometric progression of gamma (<Latex>\gamma</Latex>)
    the discounted returns approach 0 when they are far into the future.
  </p>
  <p>
    The interactive example below show how the discounting rate progresses
    towards 0 (y-axis) depending on timesteps (x-axis) and the gamma. The lower
    the gamma the faster the progression.
  </p>
  <Discounting />
  <div class="separator" />

  <h2>Policy</h2>
  <p>
    A policy is a mapping from states to either actions directly or a
    probability of an action, depending on whether the policy is deterministic
    or stochastic.
  </p>
  <p>
    If a policy is <em>deterministic</em> we define a policy as a mapping from
    state <Latex>s</Latex> to action <Latex>a</Latex>. In that case the notation
    that we use for policy is <Latex>{String.raw`\mu(s)`}</Latex>. An action <Latex
      >A_t</Latex
    > is generated by using the state
    <Latex>S_t</Latex> as the input into the policy function: <Latex
      >{String.raw`A_t = \mu(S_t)`}</Latex
    >.
  </p>
  <p>
    If a policy is <em>stochastic</em> we define a policy as a mapping from a
    state <Latex>s</Latex> to a probability of an action <Latex>a</Latex>, where
    the mathematical notation is <Latex
      >{String.raw`\pi{(a \mid s)} = Pr[A_t = a \mid S_t = s]`}</Latex
    >
    . This notation can theoretically also be applied to a deterministic case, where
    policy <Latex>{String.raw`\pi{(a \mid s) = 1}`}</Latex> for for the selected
    action and <Latex>{String.raw`\pi{(a \mid s) = 0}`}</Latex> for the rest of the
    legal actions. To generate an action using a stochastic policy we define <Latex
      >{String.raw`\pi{(. \mid S_t)}`}</Latex
    >
    to be the distribution of actions given states, where actions are draws from
    a policy distribution <Latex>{String.raw`A_t \sim \pi{(. \mid S_t)}`}</Latex
    >.
  </p>
  <div class="separator" />

  <h2>Value Functions</h2>
  <p>
    In reinforcement learning we often deal with stochastic environments and
    with stochastic policies. That stochasticity produces different trajectories
    and therefore different returns <Latex>G_t</Latex>
    even when the starting state <Latex>S_0</Latex> and the policy <Latex
      >{String.raw`\pi`}</Latex
    >
    remain the same. But how can we measure how good it is for the agent to use a
    certain policy <Latex>\pi</Latex>, if the generated returns are not
    consistent? By using value functions!
  </p>
  <p>
    For a given policy <Latex>\pi</Latex> value functions map states or state-action
    pairs to “goodness” values, where goodness is calculated as the expected sum
    of rewards and higher values mean more favorable states or state-action pairs.
  </p>
  <p class="info">
    In simple words value functions calculate the following: How much return <Latex
      >G_t</Latex
    > does the agent expect to receive when it faces the state <Latex>S_t</Latex
    > and always follows the policy <Latex>\pi</Latex>.
  </p>
  <p>
    The state-value function <Latex>{String.raw`v_{\pi}(s)`}</Latex> calculates the
    expected return when the agent is in the state <Latex>s</Latex>
    and follows the policy <Latex>{String.raw`\pi`}</Latex>.
  </p>
  <p>
    <em>State-Value Function:</em>
    <Latex>{String.raw`v_{\pi}(s) = \mathbb{E_{\pi}}[G_t \mid S_t = s]`}</Latex>
  </p>
  <p>
    The action-value function <Latex>{String.raw`q_{\pi}(s, a)`}</Latex> calculates
    the expected return when the agent faces the state <Latex>s</Latex>, takes
    the action <Latex>a</Latex> at first and then keeps following the policy <Latex
      >{String.raw`\pi`}</Latex
    >.
  </p>
  <p>
    <em>Action-Value Function:</em>
    <Latex
      >{String.raw`q_{\pi}(s, a) = \mathbb{E_{\pi}}[G_t \mid S_t = s, A_t = a]`}</Latex
    >
  </p>
  <div class="separator" />

  <h2>Optimality</h2>
  <p>
    At the beginning of the chapter we asked ourselves what it means for an
    agent to solve a Markov decision process. The solution of the MDP means that
    the agent has learned an optimal policy function.
  </p>
  <p class="info">
    For the agent to solve the Markov decision process means to find the optimal
    policy.
  </p>
  <p>
    Optimality implies that there is a way to compare different policies and to
    determine which of the policies is better. In a Markov decision process
    value functions are used as a metric of the goodness of a policy. The policy <Latex
      >{String.raw`\pi`}</Latex
    >
    is said to be better than the policy <Latex>{String.raw`\pi`}</Latex> if and
    only if the value function of <Latex>{String.raw`\pi`}</Latex> is larger or equal
    to the value function of policy <Latex>{String.raw`\pi'`}</Latex> for all states
    in the state set <Latex>{String.raw`\mathcal{S}`}</Latex>.
  </p>
  <p>
    <Latex>{String.raw`\pi \geq \pi’`}</Latex>
    if and only if <Latex>{String.raw`v_{\pi}(s) \geq v_{\pi'}(s)`}</Latex>
    for all <Latex>{String.raw`s \in \mathcal{S}`}</Latex>
  </p>
  <p>
    The optimal policy <Latex>{String.raw`\pi_*`}</Latex> is the policy that is better
    (or at least not worse) than any other policy.
  </p>
  <p>
    <Latex>{String.raw`\pi_* \geq \pi`}</Latex> for all <Latex
      >{String.raw`\pi`}</Latex
    >
  </p>
  <p>
    The state-value function and the action-value function that are based on the
    optimal policy are called optimal state-value and optimal action-value
    function respectively.
  </p>
  <p>The optimal state-value funtion:</p>
  <p>
    <Latex>{String.raw`v_*(s) = \max_{\pi} v_{\pi}(s)`}</Latex> for all states <Latex
      >{String.raw`s \in \mathcal{S}`}</Latex
    >
  </p>
  <p>The optimal action-value function:</p>
  <p>
    <Latex>{String.raw`q_*(s, a) = \max_{\pi} q_{\pi}(s, a)`}</Latex> for all states
    <Latex>{String.raw`s \in \mathcal{S}`}</Latex> and all actions <Latex
      >{String.raw`a \in \mathcal{A}`}</Latex
    >
  </p>
  <p class="info">
    There might be several optimal policies, but there is always only one
    optimal value function.
  </p>
  <div class="separator" />

  <h2>Bellman Equations</h2>
  <p>
    An important property of returns is that they can be expressed in terms of
    future returns to write a more compact definition.
  </p>
  <Latex
    >{String.raw`
      \begin{aligned}
      G_t & = R_{t+1} + \gamma{R_{t+2}} + \gamma^2{R_{t+3}} + … \\
      & = R_{t+1} + \gamma{(R_{t+2} + \gamma{R_{t+3}} + ...)} \\
      & = R_{t+1} + \gamma{G_{t+1}}
      \end{aligned}`}
  </Latex>

  <p>
    By using those properties we can arrive at recursive equations of value
    functions, where a value of a state can be expressed in terms of future
    state values.
  </p>
  <p>Bellman equation for the state-value function</p>
  <Latex
    >{String.raw`
      \begin{aligned}
      v_{\pi}(s) & = \mathbb{E_{\pi}}[G_t \mid S_t = s] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]
      \end{aligned}
  `}</Latex
  >
  <p>Bellman equation for the action-value function</p>
  <Latex
    >{String.raw`
      \begin{aligned}
      q_{\pi}(s, a) & = \mathbb{E_{\pi}}[G_t \mid S_t = s, A_t = a] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = a]
      \end{aligned}
  `}</Latex
  >
  <p>
    Equations of the above form are called Bellman equations, named after the
    mathematician Richard E. Bellman. At the very first glance it might not seem
    like the equations add additional benefit to the definition of value
    functions, but the recursive relationships is what makes many of the
    reinforcement learning algorithms work. Think for example about the return <Latex
      >G_t</Latex
    > in a continuing environment. The definition of <Latex>G_t</Latex> implies that
    we play for an infinite amount of timesteps before we can calculate a single
    return. Bellman equations on the other hand will allow us to work with estimates
    and to use a return from a single timestep to make improvements.
  </p>
  <p>
    Using the same recursive relationsships we can redefine optimal value
    functions.
  </p>
  <p>Bellman Optimality Equation for the state-value function:</p>
  <Latex
    >{String.raw`
      \begin{aligned}
      v_*(s) & = \max_{a} q_{{\pi}_*}(s, a) \\
      & = \max_{a} \mathbb{E_{\pi_{*}}}[G_t \mid S_t = s, A_t = a] \\
      & = \max_{a} \mathbb{E_{\pi_{*}}}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\
      & = \max_{a} \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]
      \end{aligned}
  `}</Latex
  >

  <p>Bellman Optimality Equation for the action-value function:</p>
  <Latex
    >{String.raw`
      \begin{aligned}
      q_*(s, a) & = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] \\
      & = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a]
      \end{aligned}
  `}</Latex
  >
  <div class="separator" />
</Container>
