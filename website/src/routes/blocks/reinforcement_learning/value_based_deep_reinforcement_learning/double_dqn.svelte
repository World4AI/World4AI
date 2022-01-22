<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Code from "$lib/Code.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Double DQN</title>
  <meta
    name="description"
    content="The double deep Q-network (DQN) is an improvemnt of DQN. Similar to double Q-learning, the agent utilizes two value functions: one for action selection and the other for value calculation. The implementation reduces the overestimation bias."
  />
</svelte:head>

<h1>Double DQN</h1>
<Question>What does a double DQN deal with the overestimation bias?</Question>
<div class="separator" />

<p class="info">
  "We first show that the recent DQN algorithm, which combines Q-learning with a
  deep neural network, suffers from substantial overestimations in some games in
  the Atari 2600 domain. We then show that the idea behind the Double Q-learning
  algorithm, which was introduced in a tabular setting, can be generalized to
  work with large-scale function approximation."
</p>
<div class="separator" />

<p>
  The DQN algorithm suffers from a similar maximization bias that is present in
  tabular Q-learning. The output of Q-functions is an estimate that might
  contain some noise. The noise that produces the highest number will be
  preferred in a max operation, even if the true action values are equal. The
  researchers at DeepMind showed that applying double learning to the DQN
  algorithms improves the performance of the agent for Atari games. This gives
  rise to double DQN (DDQN).
</p>

<p>
  In the DQN algorithm the target value is calculated by utilizing the neural
  network with frozen weights <Latex>{String.raw`\mathbf{w}^-`}</Latex>.
</p>
<Latex
  >{String.raw`
r + \gamma \max_{a'} Q(s', a', \mathbf{w}^-)
  `}</Latex
>

<p>
  It is noticeble that the same Q-function <Latex
    >{String.raw`Q(s, a, \mathbf{w}^-)`}</Latex
  > is used to select the next action <Latex>a'</Latex> and to calculate the ation-value.
  This is consistent with the classical definition of Q-learning. In double Q-learning
  two separate acion value functions are used. One is used for action selection while
  the other is used for the calculation of the target value. Using the same approach
  in DQN would not be efficient, as it would require the training of two action value
  function. However the original DQN algorithm already uses two action value funcitons:
  the neural network that is used to estimate the action values and the neural network
  with frozen weights used for bootsrapping. The two functions will allow us to to
  separate action selection and action value calculation.
</p>
<Latex
  >{String.raw`
    MSE = \mathbb{E}_{(s, a, r, s', t) \sim U(D)}[(r + \gamma Q(s', \arg\max_{a'} Q(s', a', \mathbf{w}), \mathbf{w}^-) - Q(s, a, \mathbf{w}))^2]
  `}</Latex
>

<p>
  The action in the next state is selected by utilizing the action value
  function <Latex>{String.raw`Q(s, a, \mathbf{w})`}</Latex>, while the
  calculation of the action value is performed using the frozen weights <Latex
    >{String.raw`Q(s, a, \mathbf{w}^-)`}</Latex
  >.
</p>

<p>
  The difference in the implementation between DQN and double DQN (DDQN) seems
  almost trivial in PyTorch.
</p>
<Code
  code={`
with torch.no_grad():
    target = rewards + self.gamma * self.Q_target(next_obss).max(dim=1, keepdim=True)[0] * (1 - dones)
  `}
/>

<Code
  code={`

with torch.no_grad():
    next_actions = self.Q(next_obss).max(dim=1, keepdim=True)[1].long()
    target = rewards + self.gamma * self.Q_target(next_obss).gather(dim=1, index=next_actions) * (1 - dones)
  `}
/>
