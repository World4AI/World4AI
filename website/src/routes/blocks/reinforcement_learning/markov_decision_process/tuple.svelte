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
    content="A Markov decision process is a tuple that contains the state space, the action space, the transition probabilities matrix and the reward."
  />
</svelte:head>

<h1>MDP as Tuple</h1>
<Question
  >How can you define a Markov decision process in a mathematical manner?</Question
>
<div class="separator" />

<p>
  The formal definition of a Markov decison process deals with the individual
  components of the MDP. A Markov decision process can be defined as a tuple <Latex
    >{String.raw`(\mathcal{S, A}, P, R)`}</Latex
  >, where each individual component of that tuple is a required component of a
  valid MDP. The definition of each component is going to be important in many
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
<p class="info"><Latex>P</Latex> is the transition probability function.</p>
<div class="separator" />

<h2><Latex>r</Latex>: Reward Function</h2>
<p>
  The reward function <Latex>R</Latex> calculates the reward at timestep <Latex
    >t+1</Latex
  > given state <Latex>s</Latex>
  ,action <Latex>a</Latex> and the next state <Latex>s'</Latex>.
</p>
<Latex>{String.raw`R(s,a, s') = R_{t+1}`}</Latex>
<p class="info"><Latex>R</Latex> is the reward function.</p>
<div class="separator" />
