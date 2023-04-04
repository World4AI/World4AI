<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Alert from "$lib/Alert.svelte";

  import Slider from "$lib/Slider.svelte";
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";

  // table
  import Table from "$lib/base/table/Table.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  import Mdp from "../Mdp.svelte";
  import Sequence from "../Sequence.svelte";

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

  let gamma = 0.95;
  let maxX = 100;
  let discountData = [];

  function createDiscountData() {
    discountData = [];
    for (let x = 0; x <= maxX; x++) {
      let point = { x, y: gamma ** x };
      discountData.push(point);
    }
  }
  $: gamma && createDiscountData();
</script>

<svelte:head>
  <title>Markov Decision Process Solution - World4AI</title>
  <meta
    name="description"
    content="A Markov decision process is considered to be solved once the agent found the optimal policy and the optimal value function. In essence that means that the agent has to maximize the expected sum of rewards."
  />
</svelte:head>

<Container>
  <h1>Solution to a Markov Decision Process</h1>
  <div class="separator" />
  <p>
    The <Highlight>reward hypothesis</Highlight> states that the goal of the environment
    is encoded in the rewards. If you want the agent to learn how to play chess for
    example you give out a reward of +1 when the agent wins, a reward of -1 when
    the agent loses and a reward of 0 whenever there is a draw. The goal of the agent
    is therefore to collect as many rewards as possible.
  </p>
  <p>
    We can express the same idea mathematically by defining the goal <Latex
      >G_t</Latex
    >, also called the <Highlight>return</Highlight>, as the sum of rewards from
    the timestep <Latex>t+1</Latex> to either the terminal state
    <Latex>T</Latex> if we are dealing with episodic tasks or to infinity if we are
    dealing with continuing tasks.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`G_t = R_{t+1} + R_{t+2} + R_{t+3} + R_{t+4} + \dots`}</Latex
    >
  </div>
  <p>
    To demonstrate this idea with a simple example let's once again take the MDP
    from the previous section. We are dealing with two states and an agent that
    takes random actions. Whenever the agent lands in the state 1 it receives a
    reward of 1, other wise it receives a negative reward of -1.
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
    In the example below we calculate the return from the perspective of the
    initial state <Latex>G_0</Latex>. Each time the agent receives a reward we
    add the value to the return value.
  </p>
  <Sequence f={markovDecisionProcess()} showReturn={true} />
  <p>
    In order to reduce the value of future rewards we discount each reward, by
    applying a discount factor <Latex>\gamma</Latex>, a value between 0 and 1.
    The farther away the reward is from the current time step <Latex>t</Latex>,
    the lower its contribution to the return <Latex>G_t</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots`}</Latex
    >
  </div>
  <p>
    The easiest way to illustrate this concept is by thinking about the time
    value of money. If we have to choose between getting 1000$ now and getting
    1000$ in 10 years, we should definetely choose the 1000$ in the present. The
    money could be invested for a ten year period, such that in 10 years the
    investor gets back an amout that is larger than 1000$. A reward now is
    always more valuable thant a future reward.
  </p>
  <p>
    Discounting is especially important for continuing tasks. If the episode has
    no natural ending, the return <Latex>G</Latex> could theoretically become infinite.
    Discounted rewards on the other hand approach 0 when they are far into the future.
  </p>
  <p>
    The interactive example below shows how the discounting rate dependings on
    the value of the <Latex>\gamma</Latex>. The lower the gamma the faster the
    progression towards 0.
  </p>
  <Plot
    maxWidth={700}
    domain={[0, 100]}
    range={[0, 1]}
    padding={{ top: 10, right: 10, bottom: 40, left: 40 }}
  >
    <Ticks
      xTicks={[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
      xOffset={-15}
    />
    <Ticks yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]} yOffset={15} />
    <XLabel text="Steps" fontSize={12} />
    <YLabel text="Discount Factor" fontSize={12} />
    <Path data={discountData} />
  </Plot>
  <Slider
    bind:value={gamma}
    min={0.9}
    max={0.999}
    step={0.001}
    label="Discount Factor"
    showValue={true}
  />
  <p>
    The return is very tighly integrated with the strategy thate the agent is
    following. This strategy is called policy. A better policy will obviously
    generate better returns and vice versa.
  </p>
  <Alert type="info">
    A policy is a mapping from a state to a probability of an action.
  </Alert>
  <p>
    A policy <Latex>\pi</Latex> as a mapping from a particular state
    <Latex>s</Latex>
    to a probability of an action <Latex>a</Latex>: <Latex
      >{String.raw`\pi{(a \mid s)} = Pr[A_t = a \mid S_t = s]`}</Latex
    >. To take an action in a state the agent simply draws an action from the
    policy distribution <Latex>{String.raw`A_t \sim \pi{(. \mid S_t)}`}</Latex>.
  </p>
  <p>
    In the above example we are dealing with 2 possible states and 2 possible
    actions. We can therefore simply store the policy in a mapping table. In
    future chapters we will use neural networks for more complex MDPs.
  </p>
  <Table>
    <TableHead>
      <Row>
        <HeaderEntry>State</HeaderEntry>
        <HeaderEntry>Action</HeaderEntry>
        <HeaderEntry>Probability</HeaderEntry>
      </Row>
    </TableHead>
    <TableBody>
      <Row>
        <DataEntry>0</DataEntry>
        <DataEntry>A</DataEntry>
        <span class="bg-blue-200 rounded-full font-bold">
          <DataEntry>0.5</DataEntry>
        </span>
      </Row>
      <Row>
        <DataEntry>0</DataEntry>
        <DataEntry>B</DataEntry>
        <span class="bg-blue-200 rounded-full font-bold">
          <DataEntry>0.5</DataEntry>
        </span>
      </Row>
      <Row>
        <DataEntry>1</DataEntry>
        <DataEntry>A</DataEntry>
        <span class="bg-blue-200 rounded-full font-bold">
          <DataEntry>0.5</DataEntry>
        </span>
      </Row>
      <Row>
        <DataEntry>1</DataEntry>
        <DataEntry>B</DataEntry>
        <span class="bg-blue-200 rounded-full font-bold">
          <DataEntry>0.5</DataEntry>
        </span>
      </Row>
    </TableBody>
  </Table>
  <p>
    The goal of the agent is to find a policy that maximizes returns, but simply
    maximizing the return is tricky, because in reinforcement learning we often
    deal with stochastic policies and environments. That stochasticity produces
    different trajectories and therefore different returns <Latex>G_t</Latex>.
    But how can we measure how good it is for the agent to use a certain policy <Latex
      >\pi</Latex
    > if the generated returns are not consistent? By utilizing <Highlight
      >value functions</Highlight
    >.
  </p>
  <Alert type="info"
    >A value function measures the expected value of returns.</Alert
  >
  <p>
    The state-value function <Latex>{String.raw`v_{\pi}(s)`}</Latex> calculates the
    expected return when the agent is in the state <Latex>s</Latex>
    and follows the policy <Latex>{String.raw`\pi`}</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex>{String.raw`v_{\pi}(s) = \mathbb{E_{\pi}}[G_t \mid S_t = s]`}</Latex>
  </div>
  <p>
    The action-value function <Latex>{String.raw`q_{\pi}(s, a)`}</Latex> calculates
    the expected return when the agent is the state <Latex>s</Latex>, takes the
    action <Latex>a</Latex> at first and then keeps following the policy <Latex
      >{String.raw`\pi`}</Latex
    >.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`q_{\pi}(s, a) = \mathbb{E_{\pi}}[G_t \mid S_t = s, A_t = a]`}</Latex
    >
  </div>
  <p>
    For the time being we do not have the tools to exactly calculate the value
    function for each state, but we can make an educated guess. Let's take the
    same Markov decision we have been dealing so far, but only focus on the
    state 0.
  </p>
  <p>
    The random policy that we have been using so far is clearly not optimal.
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
  <p>
    The below agent implements a policy that tends to take the action B more
    often. This behaviour will generate a higher expected return.
  </p>
  <Mdp
    config={{
      root: "0",
      states: ["0", "1", "0", "1"],
      actions: ["A", "B"],
      rewards: ["-1", "5", "-1", "5"],
      actionp: [0.2, 0.8],
      p: [0.7, 0.3, 0.2, 0.8],
      type: "decision",
    }}
  />
  <p>
    A state-value function allows the agent to assign a goodness value to each
    of the states for a given policy. Change the policy and observe if you see
    any improvements. That is where the action-value function comes into play.
    The action-value function allows us to reason about changing a single
    action. We can ask the question what would happen if we generally keep the
    same policy and only change the next action. Would that new policy be more
    benefitial to us? If yes, maybe we should change our policy. And the goal of
    the agent is to find the optimal policy.
  </p>
  <Alert type="info">
    For the agent to solve the Markov decision process means to find the optimal
    policy.
  </Alert>
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

  <div class="flex justify-center">
    <Latex
      >{String.raw`\pi \geq \pi \iff
   v_{\pi}(s) \geq v_{\pi'}(s) 
    \text{ for all } s \in \mathcal{S}`}</Latex
    >
  </div>
  <p>
    The optimal policy <Latex>{String.raw`\pi_*`}</Latex> is therefore the policy
    that is better (or at least not worse) than any other policy.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\pi_* \geq \pi \text{ for all }
     \pi`}</Latex
    >
  </div>
  <p>
    It is not hard to imagine the optimal strategy for the two-states MDP. The
    agent needs to take the action B as often as possible. This action increases
    the probability to land in the state 2 and thus increases the probability of
    getting a reward of 5. In other words always taking the actio B would
    maximize the value function.
  </p>
  <Mdp
    config={{
      root: "0",
      states: ["0", "1", "0", "1"],
      actions: ["A", "B"],
      rewards: ["-1", "5", "-1", "5"],
      actionp: [0, 1],
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
      actionp: [0, 1],
      p: [0.9, 0.1, 0.3, 0.7],
      type: "decision",
    }}
  />
  <p>
    How we can numerically find the optimal policy <Latex>\pi_*</Latex> and the corresponding
    value functions <Latex>v_*</Latex>, <Latex>q_*</Latex> is going to be the topic
    of the rest of the reinforcement learning block. The essence of reinforcement
    learning is to find the optimal policy, given that the environment contains a
    Markov decision process under the hood.
  </p>
  <div class="separator" />
</Container>
