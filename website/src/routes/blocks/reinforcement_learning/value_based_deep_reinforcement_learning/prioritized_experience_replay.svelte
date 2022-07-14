<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import MemoryBuffer from "./_dqn/MemoryBuffer.svelte";
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Prioritized Experience Replay</title
  >
  <meta
    name="description"
    content="The prioritized experience replay (PER) adjust the memory buffer in such a way, that assigns each experience tuple a priority. Experience tuples with higher priorities are used more often for training."
  />
</svelte:head>

<h1>Prioritized Experience Replay</h1>
<Question
  >What problem does the prioritized experience replay (PER) solve?</Question
>
<div class="separator" />

<p class="info">
  "In this paper we develop a framework for prioritizing experience, so as to
  replay important transitions more frequently, and therefore learn more
  efficiently.
</p>
<div class="separator" />

<p>
  The experience replay is one of the major DQN components that make the
  algorithm so efficient. The agent is able to store already seen experiences
  and to reuse them several times in training before they are discarded. Each of
  the past experiences has the same chance of being drawn and being used for
  training.
</p>
<MemoryBuffer prioritized={false} />
<p>
  But the uniform distribution with with the experiences are drawn is also the
  drawback of the experience replay. It is reasonable to assume that some
  experiences are more important and therefore better suited to learn from. This
  is where the prioritized experience replay (PER) comes into play. Each of the
  experience tuples has a priority assigned to it and the probability with which
  the tuple is likely to be drawn from the replay buffer and be used in training
  increases with higher priority. That way more important experiences are used
  more often and contribute to faster learning of the agent.
</p>
<p>
  The interactive example below is an attempt to visualize to procedure of the
  prioritized experience replay technique. You will notice that each of the
  experiences in the memory buffer have a different size. This size represents
  the priority of the experiece and has therefore a higher probability to be
  drawn. Each time an experience is drawn, the priority is reduced, because we
  generally expect that the agent has already learned from that particular
  experience. Additionally we can observe that the newest experiences have
  generally extremely high priority. This is because each new experience gets
  automatically a the highest priority to make sure that it is used at least
  once for learning.
</p>
<MemoryBuffer prioritized={true} />
<p>
  The creators of PER mention, that the ideal quantity of priority would be a
  measure of how much an agent can learn from a given experience. Such a measure
  is obviously not available and TD error is used as a proxy of priority.
</p>
<Latex
  >{String.raw`\delta = r + \gamma \max_{a'} Q(s', a', \mathbf{w}^-) - Q(s, a, \mathbf{w})`}</Latex
>
<p>
  This is obviously not an ideal measure, because the TD error is based on a
  noisy estimate, but we will assume that this is valid approximation of the
  importance of an experience tuple.
</p>
<p>
  TD error can be positive or negative, but we are only interested in the
  magnitude of the error and not in the direction. Therefore the absolute value
  of the error, <Latex>| \delta |</Latex>, is going to be used.
</p>
<p>
  If we sampled only according to the magnitude of TD error, some experiences
  would not be sampled at all before they are discarded. This is especially
  problematic when we consider that we bootstrap and have only access to
  estimates of TD errors. Therefore we generally want to sample experiences with
  high TD error more often, but still have a non zero probability for
  experiences with low TD errors. DeepMind proposes two approaches to calculate
  priorities.
</p>
<p>
  Proportilan priorization: <Latex>p_i = |\delta_i| + \epsilon</Latex>, where <Latex
    >\epsilon</Latex
  > is a positive constant that makes sure that experience tuples with a TD error
  of 0 still have a non-zero percent probability of being selected.
</p>
<p>
  Ranked-based priorization: <Latex>{String.raw`p_i = \frac{1}{rank(i)}`}</Latex
  >, where rank(i) is the index number of an experience tuple in a list, in
  which all absolute TD errors are sorted in descending order. Ranked-based
  prioritization is expected to be less sensitive to outliers, therefore this
  approach is going to be utilized in this chapter.
</p>
<p>
  The distribution is not only determined by the priority
  <Latex>p_i</Latex>, but is additionally controlled by a constant
  <Latex>\alpha</Latex>. If <Latex>\alpha</Latex> is 0 we are essentially facing
  a uniform distribution. Higher numbers of <Latex>\alpha</Latex> increase the to
  higher importance of priorities.
</p>
<Latex>{String.raw`P(i) = (p^{\alpha}_i) / (\sum_k p^{\alpha}_k)`}</Latex>
<p>
  Measuring TD errors for all experience tuples at each time step would be
  extremely inefficient, therefore the updates are done only periodically. The
  TD errors are updated only once they are drawn from the memory buffer and used
  in the training step. This is due to the fact that TD errors have to be
  calculated at the training step anyway and no additional computational power
  is therefore required. The calculations are not done for new experiences
  therefore each new experience tuple will receive the highest possible
  priority.
</p>
<p>
  If we are not careful and keep using the prioritized experience replay without
  any adjustment to the update step, we will introduce a bias. Let us assume
  that we possess the weights of the policy that minimize the mean squared error
  for the optimal policy. We utilize the policy and interact with the
  environment to fill the replay buffer. Lastly we want to recreate the weights
  for the above mentioned policy using the filled replay buffer. If we use the
  prioritized experience replay we utilize a different distribution than the one
  that is implied by the optimal weights, which is the uniform distribution. For
  example we might draw rare experiences more often, which would imply gradient
  descent steps calculated based on rare experiences more often. On the one hand
  we want to use important experiences more often, but we would also like to
  avoid the bias, especially in the long run. For that purpose we adjust the
  gradient descent step by a weight factor.
</p>
<Latex>{String.raw`w_i = (N \cdot P(i))^{-\beta}`}</Latex>

<p>
  The simplest way to imagine why the adjustment works is to imagine that we
  have uniform distribution.<Latex>{String.raw`{P(i)} `}</Latex> becomes
  <Latex>{String.raw`{\frac{1}{N}}`}</Latex> and the whole expression amounts to
  1, indicating that the uniform distribution is already the correct one and we do
  not need any adjustments. The <Latex>\beta</Latex> factor is used to control the
  correction factor. The requirement that we would like to impose is the uniform
  distribution at the end of the training. Therefore we start with a low <Latex
    >\beta</Latex
  >
  and allow for stronger updates towards the rare experiences and increase the value
  over time to make full corrections.
</p>
