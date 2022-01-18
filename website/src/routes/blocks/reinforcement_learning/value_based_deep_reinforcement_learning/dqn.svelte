<script>
  import Question from "$lib/Question.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Latex from "$lib/Latex.svelte";
  import CartPole from "$lib/reinforcement_learning/CartPole.svelte";
  import MemoryBuffer from "./_dqn.svelte/MemoryBuffer.svelte";
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
<h2>Divergence</h2>
<p>
  In tabular Q-learning each experience is thrown away as soon as it has been
  used for training. The tabular Q-function contains all the required
  information to arrive at the optimal value function. If we apply the same
  logic to approximative value functions, that would imply that we would take a
  single gradient descent step after taking a single action and throw that
  experience away.
</p>
<Latex
  >{String.raw`
  w_{t+1} \doteq w_t - \frac{1}{2}\alpha\nabla[r + \gamma \max_{a'} \hat{Q}(s', a', \mathbf{w}_t) - \hat{Q}(s, a, \mathbf{w}_t)]^2
  `}</Latex
>
<p>
  This naive implementation of deep Q-learning will most likely diverge. We
  might observe improvements until a breaking point, when the neural network
  will start failing catastrophically, never to recover again. In some cases we
  would be able to achieve descent results in simple environments like the cart
  pole environment, but achieving good results for Atari games is almost
  impossible with the naive above implementation.
</p>
<div class="separator" />
<h2>Experience Replay</h2>

<p>
  One of the reasons for the divergence of value based deep reinforcement
  learning algorithms is the high correlation of sequential observations. When
  we use gradient descent at each timestep we need to assume, that the current
  observation of the environment is very close to the previous observation. To
  convince yourself you can look at the example of the cart pole above. The cart
  position does not jump from 0 to 1, but moves slowly in incremental steps.
  From supervised learning we know, that the gradient descent algorithm assumes
  the data to be i.i.d. (independently and identically distributed), which is a
  hard limitation given the sequential nature of reinforcement learning. Thus
  sequential observations contribute to the destabilization of the learning
  process.
</p>
<p>
  Additionally we should also remember that in supervised learning, we rarely
  use stochastic gradient descent, instead we use batch gradient descent. This
  has the advantage of reducing the variance of a single observation and of
  making the calculations more efficient through parallel processing on the GPU.
</p>
<p>
  The experience replay technique uses a data structure called memory buffer to
  alleviate the above problem. Each experience tuple <Latex
    >{String.raw`e_t = (s_t, a_t, r_t, s_{t+1}, d_t)`}</Latex
  > is stored in the memory buffer: a data structure with limited capacity. At each
  time step the agent faces a certain observation, uses epsilon-greedy action selection
  and collects the corresponding reward and next observation. The whole tuple is
  pushed into the memory buffer <Latex
    >{String.raw`D_t = \{e_1, ... , e_t\}`}</Latex
  >. At full capacity the memory buffer removes the oldest tuple to make room
  for the new experience.
</p>
<p>
  Before the agent starts optimizing the value function, there is usually a so
  called warm-up phase. In that phase the agent collects random experience, in
  order to fill the memory buffer with a minimum amount of tuples.
</p>
<p>
  The agent learns only from the collected experiences and never online. At each
  time step the agent gets a randomized batch from the memory buffer and uses
  the whole batch to apply batch gradient descent. Using experience replay the
  mean squared error can be defined as follows.
</p>

<Latex
  >{String.raw`MSE \doteq \mathbb{E}_{(s, a, r, s', d) \sim U(D)}[(r + \gamma \max_{a'} Q(s', a', \theta) - Q(s, a, \theta))^2]`}</Latex
>
<p>
  We provide the interactive example of the experience replay below. We suggest
  you play with it in order to get a better intuitive understanding. The batch
  size corresponds to 3, the warmup phase is 6 and the maximum length of the
  buffer is 15. When you take a step a new experience is collected and the batch
  is chosen randomly from the available experiece tuples in the memory buffer.
  The oldest tuple is thrown away once the maximum size is reached.
</p>
<MemoryBuffer />
<p>
  The maximum length of the buffer and the batch size depend on the task the
  agent needs to solve. In the original implementation by DeepMind the memory
  size corresponded to 1,000,000 and batch size to 32. Depending on your
  hardware you might need to reduce the memory size. Especially for simpler
  tasks a memory size between 10,000 and 100,000 is usually sufficient.
</p>

<div class="separator" />
<h2>Frozen Target Network</h2>
<div class="separator" />
<h2>Loss Function</h2>
<div class="separator" />
<h2>Atari</h2>
<h3>Architecture</h3>
<h3>Preprocessing</h3>
<div class="separator" />
