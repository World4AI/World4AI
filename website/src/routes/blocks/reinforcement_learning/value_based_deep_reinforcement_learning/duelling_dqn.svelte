<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import CartPole from "$lib/reinforcement_learning/CartPole.svelte";

  let architectureConfig = {
    parameters: {
      0: { layer: 0, type: "input", count: 4, annotation: "Input" },
      1: { layer: 1, type: "fc", count: 7, input: [0] },
      2: { layer: 2, type: "fc", count: 5, input: [1] },
      3: { layer: 2, type: "fc", count: 5, input: [1] },
      4: {
        layer: 3,
        type: "fc",
        count: 5,
        input: [2],
        color: "var(--main-color-2)",
      },
      5: {
        layer: 3,
        type: "fc",
        count: 5,
        input: [3],
        color: "var(--main-color-1)",
      },
      6: {
        layer: 4,
        type: "fc",
        count: 2,
        input: [4],
        color: "var(--main-color-2)",
      },
      7: {
        layer: 4,
        type: "fc",
        count: 1,
        input: [5],
        color: "var(--main-color-1)",
      },

      8: {
        layer: 5,
        input: [6, 7],
        type: "addition",
        count: 2,
        annotation: "Q(s, a)",
      },
    },
  };
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Duelling DQN</title>
  <meta
    name="description"
    content="The duelling DQN aims to imrove the vanilla implementation of the DQN by separating the action value function Q(s, a) into the two components: the advantage function and the state value function."
  />
</svelte:head>

<h1>Duelling DQN</h1>
<Question>What problem does a duelling DQN solve?</Question>
<div class="separator" />

<p class="info">
  "Our dueling network represents two separate estimators: one for the state
  value function and one for the state-dependent action advantage function."
</p>
<p>
  The neural network architectures that we have seen so far were not much
  different from typical convolutional or fully connected neural networks that
  are used in supervised learning. With the duelling DQN the architecture of the
  neural network that estimates the action value function is adjusted in a way
  that is tailored specifically to reinforcement learning problems.
</p>
<div class="separator" />

<p>
  In tabular Q-learning (and by extension DQN) we estimate the action value
  function <Latex>{String.raw`Q_{\pi}(s, a)`}</Latex> directly. The action value
  function <Latex>{String.raw`Q_{\pi}(s, a)`}</Latex> measures the value of taking
  the action <Latex>a</Latex> in state <Latex>s</Latex> and following the policy
  <Latex>\pi</Latex> afterwards. We can calculate the same function as the sum of
  two components: the state value function and the advantage function.
</p>
<p>
  The state value function
  <Latex>{String.raw`V_{\pi}(s)`}</Latex> measures the goodness of a state. It is
  the expected return when the agent follows the policy <Latex>\pi</Latex> at state
  <Latex>s</Latex>. We can define the state value function in terms of the
  action value function.
</p>
<Latex
  >{String.raw`V_{\pi}(s) = \mathbb{E}_{a \sim \pi(s)}[Q_{\pi}(s, a)]`}</Latex
>
<p>
  In this section we additionally utilze the advantage function <Latex
    >A(s,a)</Latex
  >, that measures the advantage of taking the action <Latex>a</Latex> and then following
  policy <Latex>\pi</Latex> over always following the policy <Latex>\pi</Latex>.
</p>

<Latex>{String.raw`A_{\pi}(s, a) = Q_{\pi}(s, a) - V_{\pi}(s)`}</Latex>
<p>
  In other words the action value function can be rewritten in the following
  way.
</p>
<Latex>{String.raw`Q(s, a) = V(s) + A(s, a)`}</Latex>
<NeuralNetwork config={architectureConfig} />
<p>
  The architecture of the duelling DQN uses this exact interpretation of the
  action value function. In the first layers the weights are shared between the
  state value and the advantage function. In the later layers the single state
  value is calculated separately from the advantage values. The state value (red
  path) is finally added to advantage values (blue path) to reconstruct the
  action values.
</p>
<p>
  The question that you have at this moment is probably: why is it useful to
  separate the action value function into two components? The authors of the
  duelleing DQN paper mention that the agent often faces states, where it is not
  important which action is taken and it is more important to learn how valuable
  the actual state is. In other states a single action might determine the
  outcome of a particular environment. The separation of the action value
  function into two components allows the architecture to be more aware of those
  differences.
</p>
<CartPole />
<p>
  In the cart pole environment for example you if the pole is at the 90 degree
  angle and the cart is in the middle of the screen, it does not actually matter
  what action the agent is going to take and the state should be generally
  favourable. If the cart moves too far from the center or the pole angle is
  off, the advantages between the two actions are going to be very different.
</p>
<p>
  There is still a problem if we implement the calculation of the action value
  <Latex>Q(s,a)</Latex> naively as the sum of the state value <Latex>V(s)</Latex
  > and the advantage values <Latex>A(s, a)</Latex>. Imagine we have four
  possible actions and the action value function <Latex>Q(s, a)</Latex> returns the
  following values [18, 15, 16, 16] for a particular state s. There is no unique
  solution to represent the action values as the sum of the advantages and the state
  value. If we assume that the state value is 10, then the advantages amount to [8,
  5, 6, 6]. If we assume that the state value is 16 then advantages are [2, -1, 0,
  0]. There are an unlimited number of possible combinations and therefore the training
  of the neural network is not going to be stable.
</p>
<p>
  To stabilize the training process we need to remove a degree of freedom in
  order to enforce unique solutions.
</p>
<Latex>{String.raw`Q(s, a) = V(s) + A(s, a) - \max_a A(s, a)`}</Latex>
<p>
  By using the above equation we enforce the advantage for the highest action
  value to be 0. Let us look at the same example where the action values
  correspond to [18, 15, 16, 16]. The solution that we come up with to make both
  sides of the equation equal is to set the state value to 18 and the advantages
  to [0, -3, -2, -2].
</p>
<p>
  Below is a different version of the duelling architecture. Instead of
  subtracting the maximum advantage value, we are subtructing the average.
</p>
<Latex
  >{String.raw`Q(s, a) = V(s) + A(s, a) - \frac{1}{\mathcal{|A|}} \sum_a A(s, a)`}</Latex
>
<p>
  According to the authors this version is more stable. Therefore in the PyTorch
  implementation below we show this version. We only need to adjust the model of
  the Q-function and the forward pass. The algorithm itself does not require any
  adjustments.
</p>
