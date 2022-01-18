<script>
  import Question from "$lib/Question.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Latex from "$lib/Latex.svelte";
  import CartPole from "$lib/reinforcement_learning/CartPole.svelte";
</script>

<h1>Deep Q-Network (DQN)</h1>
<Question
  >How is a deep Q-network structured and what are the benefits?</Question
>
<div class="separator" />
<p>
  The paper on the deep Q-network (often abbreviated as DQN) by DeepMind is
  regarded as one of the most seminal papers on modern reinforcement learning.
  The research that came from DeepMind showed how a combination of Q-learning
  with deep neural networks can be applied to Atari games. For many Atari games
  the DQN Agent even outperformed professional human players. The results were
  inspiring and groundbreaking in many respects, but the most important
  contribution of the paper was probably the rejuvenation of a field that seemed
  to be forgotten by the public. DQN spurred a research streak that continues up
  to this day.
</p>
<p>
  Many of the solutions by the DQN seem to show creativity and even if you are
  not a reinforcement learning enthusiast, you will most likely find the
  playthroughs of Atari games by the DQN agent to be almost magical.
</p>

<p>
  In this chapter we are going to explore the components that made the deep
  Q-network successful. We will look at how Atari games can be solved, but
  before that we are going to explore solutions to simpler OpenAI gym
  environments, because those can be solved quicker, especially if you do not
  possess a modern Nvidia graphics card.
</p>
<div class="separator" />
<h2>Architecture</h2>
<p>
  In value based deep reinforcement learning we are utilizing a neural network
  to represent the approximate action value function <Latex
    >{String.raw`\hat{Q}(s, a, \mathbf{w})`}</Latex
  >, where <Latex>s</Latex> is the state of the environment, <Latex>a</Latex> is
  the action and <Latex>{String.raw`\mathbf{w}`}</Latex> is the weight vector of
  the neural network.
</p>
<CartPole />
<p>
  If we take the cart pole environment as an example of a simple OpenAI gym
  environment, we will face a state <Latex>s</Latex> represented by a 4 dimensional
  vector.
</p>
<Latex
  >{String.raw`
\mathbf{x} \doteq 
\begin{bmatrix}
  x_1 \\
  x_2 \\ 
  x_3 \\
  x_4 
\end{bmatrix}
  `}</Latex
>
<p>
  There are two possible actions in the action space <Latex
    >{String.raw`\mathcal{A}`}</Latex
  >. The agent can choose the action to move left (action 0) or to move right
  (action 1).
</p>
<p>
  Alltogether there are 5 inputs into the neural network, the four variables
  representing the state and one variable representing the value of the chosen
  action (0 or 1). The input is processed in a series of hidden layers by
  applying linear transformations and non-linear activation functions. The
  output layer is a single neuron that represents the action value of the state
  action combination.
</p>
<NeuralNetwork layers={[5, 10, 10, 1]} />
<p>
  The architecture above is theoretically sound, but has a significant practical
  drawback. In order to improve the policy the agent needs to act greedy with
  respect to the action value function<Latex
    >{String.raw`\hat{Q}(s, a, \mathbf{w})`}</Latex
  >. The agent needs to take the action with the hightest action value and that
  implies that we need to calculate the action values for all available actions
  in order to determine which action has the highest value. Each calculation
  would require the full pass through the neural network which gets
  computationally costly with the increased complexity of the network and the
  number of possible actions.
</p>
<p>
  The below architecture on the other hand is the one that is similar in spirit
  to the solution that DeepMind ended up using for DQN.
</p>
<NeuralNetwork layers={[4, 10, 10, 2]} />
<p>
  The input of the neural network consists only of the state representation<Latex
    >{String.raw`\mathbf{x}`}</Latex
  >. The output layer has as many neurons as there are available actions in the
  environment. In the cart pole environment for example only two actions are
  available, therefore the neural network has two output neurons. With the above
  architecture only one pass through the network is required and choosing the
  action with the hightest value corresponds to taking the
  <Latex>{String.raw`\arg\max_a \hat{Q}(s, a, \mathbf{w})`}</Latex> operation.
</p>
<p />
<div class="separator" />
<h2>Experience Replay</h2>
<div class="separator" />
<h2>Frozen Target Network</h2>
<div class="separator" />
<h2>Loss Function</h2>
<div class="separator" />
<h2>Atari</h2>
<h3>Architecture</h3>
<h3>Preprocessing</h3>
<div class="separator" />
