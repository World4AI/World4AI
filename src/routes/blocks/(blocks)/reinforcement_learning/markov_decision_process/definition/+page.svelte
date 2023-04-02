<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Alert from "$lib/Alert.svelte";

  import Mdp from "./Mdp.svelte";
  import Sequence from "./Sequence.svelte";

  // stochastic processes
  function bernoulliSequence() {
    return { type: "state", value: Math.random() > 0.5 ? 1 : 0 };
  }
  function markovSequence() {
    let lastState = 0;
    let f = () => {
      let result = lastState;
      let isChange = Math.random() > 0.8 ? true : false;
      if (lastState === 0) {
        if (isChange) {
          result = 1;
        }
      } else if (lastState === 1) {
        if (isChange) {
          result = 0;
        }
      }
      lastState = result;
      return { type: "state", value: result };
    };
    return f;
  }

  function markovRewardProcess() {
    let lastState = 0;
    let lastType = "reward";
    let f = () => {
      if (lastType === "reward") {
        let result = lastState;
        let isChange = Math.random() > 0.8 ? true : false;
        if (lastState === 0) {
          if (isChange) {
            result = 1;
          }
        } else if (lastState === 1) {
          if (isChange) {
            result = 0;
          }
        }
        lastState = result;
        lastType = "state";
        return { type: "state", value: result };
      }

      if (lastType === "state") {
        lastType = "reward";
        let result;
        if (lastState === 0) {
          result = -1;
        } else if (lastState === 1) {
          result = 5;
        }
        return { type: "reward", value: result };
      }
    };
    return f;
  }

  function markovDecisionProcess() {
    let lastState = 0;
    let lastAction = "A";
    let lastType = "action";
    let f = () => {
      if (lastType === "action") {
        if (lastState === 0) {
          if (lastAction === "A") {
            lastState = Math.random() < 0.7 ? 0 : 1;
          } else if (lastAction === "B") {
            lastState = Math.random() < 0.2 ? 0 : 1;
          }
        }
        if (lastState === 1) {
          if (lastAction === "A") {
            lastState = Math.random() < 0.9 ? 0 : 1;
          } else if (lastAction === "B") {
            lastState = Math.random() < 0.3 ? 0 : 1;
          }
        }
        lastType = "state";
        return { type: "state", value: lastState };
      }

      if (lastType === "reward") {
        lastType = "action";
        lastAction = Math.random() < 0.5 ? "A" : "B";
        return { type: "action", value: lastAction };
      }

      if (lastType === "state") {
        lastType = "reward";
        let result;
        if (lastState === 0) {
          result = -1;
        } else if (lastState === 1) {
          result = 5;
        }
        return { type: "reward", value: result };
      }
    };
    return f;
  }
</script>

<svelte:head>
  <title>Markov Decision Process Definition - World4AI</title>
  <meta
    name="description"
    content="The Markov decision process (MDP) is a stochastic process with the markov property that contains an interface for an agent to take actions."
  />
</svelte:head>

<h1>Definition of a Markov Decision Process</h1>
<Container>
  <div class="separator" />
  <p>
    In reinforcement learning the agent and the environment interract with each
    other in discrete timesteps. At the beginnig of the interraction the agent
    receives the initial state of the environment
    <Latex>S_0</Latex> and produces the action <Latex>A_0</Latex>
    . Based on that action the environment transitions into a new state <Latex
      >S_1</Latex
    > and generates the corresponding reward for the agent <Latex>R_1</Latex>.
    The agent in turn reacts with the action <Latex>A_1</Latex>
    and the interaction continues. In order to be able to model this interaction,
    each environment has a so called <Highlight
      >Markov decision process</Highlight
    > under the hood. These three words allow us to fully describe how the interaction
    works.
  </p>
  <Alert type="info">
    A Markov decision process is a
    <Highlight>stochastic process</Highlight> with a <Highlight
      >Markov</Highlight
    >
    property, that provides a mechanism for the agent to make
    <Highlight>decisions</Highlight> and receive rewards.
  </Alert>
  <p>Intuitively we can define a stochastic process in the following way.</p>
  <Alert type="info">
    A stochastic process is a sequence of random variables that develops over
    time.
  </Alert>
  <p>
    Let's have a look at the Bernoulli distribution <Latex>X</Latex> for example.
    This random variable produces the value 1 with the probability <Latex
      >p</Latex
    > and the value 0 with the probability <Latex>1-p</Latex>. Let's assume that <Latex
      >p</Latex
    > eqauls to 0.5.
  </p>
  <Mdp
    config={{ root: "X", states: ["0", "1"], p: [0.5, 0.5], type: "chain" }}
  />
  <p>
    At each timestep we draw a value from the distribution and generate a
    sequence of random variables.
  </p>
  <Sequence f={bernoulliSequence} />
  <p>
    In essence we can regard the Bernoulli process as an environment with just
    two states: state 0 and state 1. The environment transitions from one state
    to the next based on the probability <Latex>p</Latex>. In this type of
    environment no actions or rewards are present. If we used the Bernoulli
    process in reinforcement learning our agent would drift from one state into
    the next state without any agency and any rewards to guide its actions. The
    stochastic process that is used for reinforcement learning is called the
    <Highlight>Markov chain</Highlight>.
  </p>
  <Alert type="info">
    A Markov chain is a stochastic process that has the Markov property.
  </Alert>
  <p>
    The Bernoulli process generates a sequence of independent random variables.
    No matter what state came before the current time step, the probability for
    the next state stays unchanged. The Markov chain on the other hand exhibits
    a so called Markov property. In simple words that means that the probability
    of transitioning into the next state depends on the current state, and the
    current state only. The whole history of past states is irrelevant.
  </p>
  <p>
    Let's have a look at an example of a Markov chain with two possible states.
  </p>
  <p>
    If the current state of the environment is 0 the environment transitions
    into the state 1 with 20% probability and remains with 80% probability in
    state 0.
  </p>
  <Mdp
    config={{ root: "0", states: ["0", "1"], p: [0.8, 0.2], type: "chain" }}
  />
  <p>
    If the current state of the environment is 1 on the other hand the
    environment transitions into the state 0 with 20% probability and remains
    with 80% probability in the state 0.
  </p>
  <Mdp
    config={{ root: "0", states: ["0", "1"], p: [0.2, 0.8], type: "chain" }}
  />
  <p>
    In other words in this example the environment tends to stay in the same
    state.
  </p>
  <Sequence f={markovSequence()} />
  <p>
    But why does it make sense to model the agent-environment interaction with a
    Markov chain? Why can't we just use for example the Bernoulli process? Think
    about a chess board for example. The state of the board does not change
    randomly from move to move. We move one piece at a time, while other pieces
    remain fairly constant. The board configuration always depends on the
    configuration from the previous time step, but only on the directly
    preceding state configuration. You do not need to know how the whole game
    developed in order to make the next move. The current positions of the chess
    pieces and your actions are sufficient to create the next state of the
    board.
  </p>
  <p>
    The <Highlight>Markov reward process</Highlight> adds a reward for to each state.
  </p>
  <p>
    In the below example whenever the agent lands in the state 0, it gets a
    negative reward of -1 and whenever the agent lands in the state 1 it gets a
    positive reward of 5. The reward can theoretically also be calculated
    stochastically, but let's keep it simple and assume a deterministic reward.
  </p>
  <Mdp
    config={{
      root: "0",
      states: ["0", "1"],
      rewards: ["-1", "5"],
      p: [0.8, 0.2],
      type: "reward",
    }}
  />
  <Mdp
    config={{
      root: "1",
      states: ["0", "1"],
      rewards: ["-1", "5"],
      p: [0.2, 0.8],
      type: "reward",
    }}
  />
  <p>
    In a Markov reward process the sequence of random variables not only
    contains states but also rewards. The process drifts randomly from one state
    into the next and gives out rewards to the agent, but the agent has no
    influence on the process whatsoever, even though the agent would prefer to
    gravitate towards the state 1 as its goal is to maximize the expected sum of
    rewards.
  </p>
  <Sequence f={markovRewardProcess()} />
  <p>
    A Markov decision process adds, as the name suggests, decisions to the
    Markov chain. In other words the agent gets some agency.
  </p>
  <Alert type="info">
    A Markov decision process is a Markov chain, such that the agent can
    partially influence the outcomes of the chain.
  </Alert>
  <p>
    Additionally to the two states we add the option for the agent to take one
    of the actions: A or B. The action A increases the likelihood to move
    towards the 0's state, while the action B increases the probability to move
    towards the state 1. For the time being let's assume that the agent takes
    random actions.
  </p>
  <Mdp
    config={{
      root: "0",
      states: ["0", "1", "0", "1"],
      actions: ["A", "B"],
      rewards: ["-1", "5", "-1", "5"],
      actionp: [0.5, 0.5],
      p: [0.7, 0.3, 0.2, 0.8],
      type: "decision",
    }}
  />
  <Mdp
    config={{
      root: "1",
      states: ["0", "1", "0", "1"],
      actions: ["A", "B"],
      rewards: ["-1", "5", "-1", "5"],
      actionp: [0.5, 0.5],
      p: [0.9, 0.1, 0.3, 0.7],
      type: "decision",
    }}
  />
  <p>
    This time around the random process does not only contain states and
    rewards, but also actions taken by the agent.
  </p>
  <Sequence f={markovDecisionProcess()} />
  <p>
    In the example above the environment has still two states, state 0 and state
    1. Landing in state 1 is more advantagous as the agent receives a reward of
    5, while landing in the 0's state generates a reward of -1. In both states
    the agent can choose between two actions. The action A is the action toward
    the state 0 and the action B is the action towards the state 1. For example
    if the agent is in the state 0 it should choose the action B to move with a
    probability of 70% to the state B and receive a reward of 5. The whole
    process is the succession of states, actions and rewards. This is the same
    mechanism, that we introduced previously as the interaction between the
    agent and the environment. The interaction is nothing more than a Markov
    chain that is made out of two elements. The first element, the environment
    is static and can not be changed in any form. The second component is the
    agent. The agent can change the probabilities of the whole chain, by
    tweaking the probabilities of actions. In the example above, we expect the
    agent to improve by increasing the probabilities to move to state 1. Ideally
    that would mean that the agent will only choose action B.
  </p>
  <p>
    Now we are ready to look at the formal definition of a Markov decision
    process. This definition deals with the four components that a MDP has to
    contain in order to be valid.
  </p>
  <Alert type="info">
    A Markov decision process can be defined as a tuple with four components: <Latex
      >{String.raw`(\mathcal{S, A}, P, R)`}</Latex
    >.
  </Alert>
  <p>
    <Latex>{String.raw`\mathcal{S}`}</Latex> is the <Highlight
      >state space</Highlight
    >, that contains all possible states of the environment. In the examples
    above we were dealing with just two states, therefore our state space would
    correspond to: <Latex>{String.raw`\mathcal{S}=[0, 1]`}</Latex>.
  </p>
  <p>
    <Latex>{String.raw`\mathcal{A}`}</Latex> is the <Highlight
      >action space</Highlight
    >, that contains all possible actions of the environment. Above we were
    dealing with the environment with two actions:
    <Latex>{String.raw`\mathcal{A}=[A, B]`}</Latex>.
  </p>
  <p>
    <Latex>P</Latex> is
    <Highlight>transition probability function</Highlight>
    that provides a probability for the next state <Latex>s'</Latex> given the current
    state <Latex>s</Latex> and the action <Latex>a</Latex>. Mathematically we
    can express that idea as:
    <Latex
      >{String.raw`P(s' \mid s, a) \doteq Pr[S_{t+1}=s' \mid S_t=s, A_t=a], \forall s, s' \in \mathcal{S}, a \in \mathcal{A}`}</Latex
    >. For example the probability to transition into the state 1 from state 0,
    given that the agent chose the action B is 80%: <Latex
      >P(1 \mid 0, B) = 0.8</Latex
    >.
  </p>
  <p>
    <Latex>R</Latex> is <Highlight>reward function</Highlight>. This function
    calculates the reward given the state <Latex>s</Latex>
    , the action <Latex>a</Latex> and the next state <Latex>s'</Latex>
    <Latex>{String.raw`R(s,a, s')`}</Latex>. For example our agent would receive
    a reward of 5, whenever it landed in state 1, <Latex>R(1, A, 1) = 5</Latex>.
  </p>
  <p>
    It does not matter if we are talking about a grid world, a game of chess,
    StarCraft or the real world. We always expect the environment to conain a
    MDP with the four components under the hood. Usually we do not have access
    to those components and can not examine them explicitly, but just knowing
    that they are there allows the agent to interract with the environement and
    to learn from those interractions.
  </p>
  <div class="separator" />
</Container>
