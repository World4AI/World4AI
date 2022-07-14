<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Proximal Policy Optimization (PPO)
  </title>
  <meta
    name="description"
    content="Proximal Policy Optimization (PPO) is a state of the art actor-critic reinforcement learning algorithm, that uses a clipped objective function to maximize expected returns."
  />
</svelte:head>

<h1>Proximal Policy Optimization (PPO)</h1>
<Question>How does PPO improve TRPO and how can we implement PPO?</Question>
<div class="separator" />

<p>
  Proximal policy optimization is the direct successor to TRPO, where we try to
  optimize the following surrogate objective.
</p>
<Latex
  >{String.raw`\large \mathbb{E}\Big[\dfrac{\pi_{\theta}}{\pi_{\theta_{old}}}\hat{A}\Big] \text{, such that } D_{KL}(\pi_{\theta} || \pi_{\theta_{old}} ) < \epsilon`}</Latex
>
<p>
  Essentially we attempt to utilize old trajectories that we collected with a
  policy <Latex>{String.raw`\pi_{\theta_{old}}`}</Latex> to optimize the new policy
  and we do it in such a way, that the difference between the two policies is not
  too large. The big challenge of TRPO is to efficiency implement the constraint
  in the optimization. We would need to utilize several mathematical tricks and can
  generally not easily implement TRPO utilizing existing "automatic differentiation"
  libraries like PyTorch or TensorFlow.
</p>
<p>
  The PPO paper introduces a clipped objective function, that tries to solve the
  same problem, but in a much easier manner.
</p>
<Latex
  >{String.raw`\mathbb{E}\Big[\min\Big(\dfrac{\pi_{\theta}}{\pi_{\theta_{old}}}\hat{A}, clip\big[\dfrac{\pi_{\theta}}{\pi_{\theta_{old}}}, 1 - \epsilon, 1 + \epsilon\big]\hat{A}\Big)\Big]`}</Latex
>
<p>
  The above objective needs some unpacking in order to grasp what is going on,
  but the overall idea is relatively simple.
</p>
<p>
  In the first step we clip the ratio between the old policy and the new policy
  between <Latex>1-\epsilon</Latex> and
  <Latex>1+\epsilon</Latex>, where
  <Latex>\epsilon</Latex> is a hyperparameter that takes a small value like 0.2.
  The clip function basically squeezes the ratio between the two values policies
  and does not allow the two policies to stray too far away from each other.
</p>
<p>
  In the second step we take the minimum between the clipped ratio multiplied
  with the advantage and the actual ratio multiplied with the advantage. To
  understand why this is necessary we need to consider two cases: the advantage <Latex
    >{String.raw`\hat{A}`}</Latex
  > is posivite and the advantage
  <Latex>{String.raw`\hat{A}`}</Latex>
  is negative. When <Latex>{String.raw`\hat{A}`}</Latex> is positive, the algorithm
  will try to push the ratio as high as possible. The <Latex>\min</Latex> operation
  will make sure that the ratio is not larger than <Latex>1+\epsilon</Latex>.
  When <Latex>{String.raw`\hat{A}`}</Latex> is negative, gradient descent will attemt
  to push the ratio as low as possible, but the min operation will make sure that
  the ratio is not lower than <Latex>1-\epsilon</Latex>, because a lower ratio
  between the two probabilities generates a higher number when the advantage is
  negative. For example <Latex>0.9 * (-1) \lt 0.8 *(-1)</Latex>.
</p>
<p>
  The two steps make sure that the probabilities are never pushed too far away
  from each other.
</p>
<div class="separator" />
