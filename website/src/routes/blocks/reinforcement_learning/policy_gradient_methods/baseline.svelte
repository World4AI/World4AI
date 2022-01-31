<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>REINFORCE With Baseline</h1>
<Question>What is a baseline and why is it useful?</Question>
<div class="separator" />

<p>
  The reinforce algorithm implemented in the last chapter is the first viable
  variant of a policy gradient method, yet not the one with the least variance.
  In this chapter we introduce an improvement to REINFORCE. The algorithm is
  called REINFORCE with baseline, but sometimes the name vanilla policy gradient
  (VPG) is also used.
</p>

<Latex
  >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k
    \end{aligned}
  `}</Latex
>

<p>
  The gradient calculation for REINFORCE can be interpreted as follows. The log
  probability of actions that generate high returns is increased, while the log
  probability of actions that generate negative returns is decreased.
</p>

<p>
  But what if all returns are positive or clustered like in the image above? The
  probability of actions with highest returns will increase more than those with
  lower returns. But the process is slow.
</p>
<Latex
  >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k - b
    \end{aligned}
`}</Latex
>

<p>
  The next improvement we are going to introduce is the baseline. The baseline
  <Latex>b</Latex> is deducted from the return, which reduces the variance.
</p>

<p>
  Intuitively some of the positive returns might stay positive, while others are
  pushed below the zero line. That makes the gradient positive for returns above
  the baseline and negative for returns below the baseline. There are different
  choices for the baseline <Latex>b</Latex>, which has an impact on how much
  variance and bias the algorithm has.
</p>

<Latex
  >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k - V(S_t^{(i)})
    \end{aligned}
  `}</Latex
>

<p>
  REINFORCE with baseline uses the state value function <Latex>V(S_t)</Latex> as
  the baseline. This makes perfect sense as only the probability of those actions
  are increased that generate returns that are above the expected sum of rewards.
  In our implementation <Latex>V(S_t)</Latex> is going to be a learned neural network,
  meaning that we have two separate functions, one for the policy and one for the
  state value. In VPG the policy and value functions are updated with monte carlo
  simulations, therefore this algorithm is only suited for episodic tasks.
</p>
