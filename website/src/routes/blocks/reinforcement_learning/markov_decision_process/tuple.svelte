<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import { gridMap } from "$lib/reinforcement_learning/grid_world/maps";

  let env = new GridEnvironment(gridMap);
  let cellsStore = env.cellsStore;
  $: cells = $cellsStore;

  import Action from "$lib/reinforcement_learning/grid_world/Action.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | MDP As Tuple</title>
  <meta
    name="description"
    content="A Markov decision process is a tuple that contains the state space, the action space, the transition probabilities matrix, the reward function and the discount factor gamma."
  />
</svelte:head>

<h1>MDP as Tuple</h1>
<Question
  >How can you define a Markov decision process in a mathematical manner?</Question
>
<div class="separator" />

<p>
  The most formal definition of a Markov decison process deals with the
  individual components of the MDP. A Markov decision process can be defined as
  a tuple <Latex>{String.raw`(\mathcal{S, A}, P, R, \gamma)`}</Latex>, where
  each individual component of that tuple is a required component of a valid
  MDP. The definition of each component is going to be important in many
  subsequent sections, because those definitions are the basis of most
  mathematical proofs in reinforcement learning, therefore in the following
  sections we will take a look at each of the contents of the tuple
  individually.
</p>
<div class="separator" />

<h2><Latex>{String.raw`\mathcal{S}`}</Latex>: State Space</h2>
<p>
  In a Markov decision process <Latex>{String.raw`\mathcal{S}`}</Latex> is the state
  space: the set that contains all possible states of the environment.
</p>
<div class="flex-center">
  <Grid {cells} showOnlyGrid={true} />
</div>
<p>
  In the example above for example we are dealing with a 5X5 grid world, where
  each state can be represented by a row and column tuple: (row, column).
  Alltogether there are exactly 25 possible states, therefore our state space
  looks as follows: <Latex
    >{String.raw`\mathcal{S}=[(0,0), (0, 1), (0, 2), ... , (4, 4)]`}</Latex
  >
  . In practice the states of a grid world are often represented by a single number
  <Latex>{String.raw`\mathcal{S}=[0, 1, 2, ... ,24]`}</Latex>. Both
  representations are equivalent, because the representaion is sufficient to
  uniquely identify the state that the agent faces.
</p>
<p />
<p class="info">
  <Latex>{String.raw`\mathcal{S}`}</Latex> is the set of all legal states.
</p>
<div class="separator" />

<h2><Latex>{String.raw`\mathcal{A}`}</Latex>: Action Space</h2>
<p>
  In a Markov decision process <Latex>{String.raw`\mathcal{A}`}</Latex> is the action
  space, that contains all possible actions of the environment.
</p>
<div class="flex-space">
  <Action action={0} />
  <Action action={1} />
  <Action action={2} />
  <Action action={3} />
</div>
<p>
  In the simple grid world environment the agent has the option to move into
  four different directions: north, east, south and west. The same actions are
  represented in the environment by four different numbers: 0, 1, 2 and 3. For
  that reason the action space in that particular grid world is <Latex
    or
    a
    barrier>{String.raw`\mathcal{A}=[0, 1, 2, 3]`}</Latex
  >
  . Even if in some states it is not possible to move into a particular direction
  (when the agent faces a wall or a barrier), the state space is usually kept consistent
  across the whole state space and the agent is expected to learn that a particulare
  action is not useful in that particular state.
</p>
<p class="info">
  <Latex>{String.raw`\mathcal{A}`}</Latex> is the set of all legal actions.
</p>
<div class="separator" />

<h2><Latex>P</Latex>: Transition Probability Function</h2>
<p>
  Each Markov decision process has a transition probabilities function <Latex
    >P</Latex
  >
  that provides a probability for the next state <Latex>s'</Latex> given the current
  state <Latex>s</Latex> and the action <Latex>a</Latex>. Mathematically we can
  express that idea as follows:
</p>
<Latex
  >{String.raw`P(s' \mid s, a) \doteq Pr[S_{t+1}=s' \mid S_t=s, A_t=a], \forall s, s' \in \mathcal{S}, a \in \mathcal{A}`}</Latex
>
<p>
  Let us assume that the environment is in the initial state and the agent
  decides to move to the east, therefore <Latex>s=(0,0), a=1</Latex>. Let us
  further assume that the environment transitions in 1/3 of cases into the
  desired direction, in 1/3 of cases to the left of the desired direction and in
  1/3 of cases to the right of the desired direction. In the initial state of <Latex
    >{String.raw`s=(0,0)`}</Latex
  >
  there is no chance to move to the left of the desired position (east), because
  the agent faces a wall towards north. In that case the agent would remain at the
  initial position. Alltogether the transition probabilities given the initial state
  and the action to move east look as follows.
</p>
<Latex>{String.raw`P(S_{t+1}=(0,1) \mid S_t=(0,0), A_t=1) = 33.33\%`}</Latex>
<Latex>{String.raw`P(S_{t+1}=(0,0) \mid S_t=(0,0), A_t=1) = 33.33\%`}</Latex>
<Latex>{String.raw`P(S_{t+1}=(1,0) \mid S_t=(0,0), A_t=1) = 33.33\%`}</Latex>
<p class="info"><Latex>P</Latex> is the transition probability function.</p>
<div class="separator" />

<h2><Latex>r</Latex>: Reward Function</h2>
<p>
  The reward function calculates the expected value of the reward given state <Latex
    >s</Latex
  >
  and action <Latex>a</Latex> at time step <Latex>t</Latex>. Mathematically this
  can be written as follows.
</p>
<Latex
  >{String.raw`R(s,a) \doteq \mathbb{E}[R_{t+1} \mid S_{t}=s,A_{t}=a]`}</Latex
>
<p>
  In the example above the state of the environment is <Latex>(3, 0)</Latex>
  . The agent selects the action 2 to move south and to hopefully receive a positive
  reward. If the agent lands on the blue square the game ends and the agent receives
  a reward of 1. In any other case the agent lands on the red square and receives
  a reward of -1. Just as in the example above the state transitions only in 1/3
  of cases in the desired direction. When the agent selects the action 2 there is
  a 33.3% chance of landing on the triangle, a 33.3% chance of not moving and a 33.3%
  chance of moving to the right. The given state and action imply a reward of approximately
  -0.33.
</p>
<Latex
  >{String.raw`R((3, 0),2) = \mathbb{E}[R_{t+1} \mid S_{t}=(3,0),A_{t}=2] = 0.33 * (-1) + 0.33 * (-1) + 0.33 * (+1) = -0.33 `}</Latex
>
<p class="info"><Latex>R</Latex> is the reward function.</p>
<div class="separator" />

<h2>Gamma: Discount Factor</h2>
<p>
  Consider the following example. You can get 1000$ now or 1000$ in 10 years.
  What would you choose? The answer is hopefully 1000$ now. The reason every
  rational agent would choose 1000$ is the time value of money or generally
  speaking the time value of rewards. In the case of dollars you could invest
  the money for 10 years and get an amount that is larger. Therefore there
  should be a compensation if the agent decides to delay his reward. The gamma <Latex
    >{String.raw`\gamma`}</Latex
  >
  (also called discount factor) is a value between 0 and 1 that is used to adjust
  the value of rewards. Future rewards are considered of less value. Usually in practice
  the value of gamma is between 0.9 and 0.99.
</p>
<p class="info">
  <Latex>{String.raw`\gamma`}</Latex> (gamma) is the discount factor, where <Latex
    >{String.raw`0 \leq \gamma \leq 1`}</Latex
  >
  .
</p>
<p>
  The value of rewards from the perspective of the agent at time step <Latex
    >t</Latex
  >
  is as following:
</p>
<p>
  The value of a reward received at timestep <Latex>t+1</Latex> is <Latex
    >{String.raw`\gamma^0 * R_{t+1}`}</Latex
  >
</p>
<p>
  The value of a reward received at timestep <Latex>t+2</Latex> is <Latex
    >{String.raw`\gamma^1 * R_{t+2}`}</Latex
  >
</p>
<p>
  The value of a reward received at timestep <Latex>t+3</Latex> is <Latex
    >{String.raw`\gamma^2 * R_{t+3}`}</Latex
  >
</p>
<p>
  Mathematically speaking if you are dealing with episodic tasks that have a
  defined ending, like the grid world environment above, discounting is not
  strictly required. For continuing tasks that can theoretically go on forever a
  discount factor is required. The reason for that is the need for the agent to
  maximize rewards. If the task is continuing then the sum of rewards might
  become infinite and the agent can not deal with that. If the value of gamma is
  between 0 and 1 then the sum becomes finite.
</p>
<p class="info">
  <strong>Episodic</strong> tasks are tasks that have a natural ending. The last
  state in an episodic task is called a <em>terminal state</em>. The letter <Latex
    >T</Latex
  >
  is used to mark the final time step.
</p>
<p class="info">
  <strong>Continuing</strong> tasks are tasks that do not have a natural ending and
  may theoretically go on forever.
</p>

<p>Let’s assume a gamma with a value of 0.9.</p>

<p>
  At <Latex>t+3</Latex>: the discount factor is <Latex
    >{String.raw`\gamma^2 = 0.81`}</Latex
  >
</p>
<p>
  At <Latex>t+5</Latex>: the discount factor is <Latex
    >{String.raw`\gamma^4 = 0.66`}</Latex
  >
</p>
<p>
  At <Latex>t+11</Latex>: the discount factor is <Latex
    >{String.raw`\gamma^{10} = 0.35`}</Latex
  >
</p>
<p>
  At <Latex>t+21</Latex>: the discount factor is <Latex
    >{String.raw`\gamma^{20} = 0.12`}</Latex
  >
</p>
<p>
  At <Latex>t+51</Latex>: the discount factor is <Latex
    >{String.raw`\gamma^{50} = 0.005`}</Latex
  >
</p>
<p>
  The discount factor keeps approaching 0, which makes the value of rewards in
  the far future almost 0. That prevents an infinite sum of rewards.
</p>
<p>
  Gamma <Latex>{String.raw`\gamma`}</Latex> is defined as a part of an MDP, but in
  practice it is treated as a hyperparameter of the agent. Theoretically gamma should
  be obvious from the environment. In practice there is no clear indiciation regarding
  how large
  <Latex>{String.raw`\gamma`}</Latex> should be. Tweaking the value might produce
  better results, because gamma can have an impact on how fast the algorithm converges
  and how stable the learning process is.
</p>
<div class="separator" />
