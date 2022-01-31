<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>REINFORCE</h1>
<Question
  >What improvement does the REINFORCE algorithm introduce and how can we
  implement the algorithm?</Question
>
<div class="separator" />

<p>
  The policy gradient algorithm developed in the previous section shows high
  variance and thus requires a high number of trajectories. In this chapter we
  are going to start to develop methods that reduce variance and implement the
  first variant of an algorithm with less variance called REINFORCE.
</p>
<div class="separator" />

<h2>Temporal Decompositoin</h2>
<Latex
  >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) R(\tau^{(i)}) \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_t^H R_t \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) [\sum_{k=0}^{t} R_k + \sum_{k=t+1}^{H} R_k]
    \end{aligned}
  `}</Latex
>

<p>
  The policy gradient is calculated by multiplying the gradient of the log
  probability of an action <Latex>A_t</Latex> given the state <Latex>S_t</Latex>
  with the return from the whole trajectory <Latex>\sum_t^H R_t</Latex>. The
  return of the trajectory can be decomposed into two distinct parts. The return
  that was already realized and can not be changed <Latex
    >{String.raw`\sum_{(k = 0)}^{t} R_k`}</Latex
  >
  and the return that is still to be earned and is going to be generated from the
  next step onward <Latex>{String.raw`\sum_{k=t+1}^{H} R_k`}</Latex>.
</p>
<Latex
  >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} R_k
    \end{aligned}
  `}</Latex
>
<p>
  There is no need to multiply the log probability of an action with the past
  return. The action has no impact on the past. It turns out that ignoring past
  returns reduces the variance.
</p>
<div class="separator" />

<h2>Discounting</h2>
<Latex
  >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k
    \end{aligned}
  `}</Latex
>
<p>
  The second adjustment to the policy gradient algorithm that we are going to
  make is the introduction of discounting. Discounting that we have already seen
  with dynamic programming and q-learning accounts for the time value of
  rewards. Additionally discounting reduces variance in policy gradients
  methods.
</p>
<div class="separator" />
