<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | REINFORCE Algorithm</title>
  <meta
    name="description"
    content="REINFORCE is one of the basic implementations of the policy gradient algorithms developed by Ronald J. Williams."
  />
</svelte:head>

<h1>REINFORCE</h1>
<div class="separator" />

<Container>
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
    probability of an action with the return from the whole trajectory <Latex
      >R(\tau)</Latex
    >. But that does not make a lot of sense, because the action at timepoint <Latex
      >t</Latex
    > can not any impact any of the rewards that were made before. We are going to
    decompose the return of the trajectory into two distinct parts. The return that
    was already realized and can not be changed <Latex
      >{String.raw`\sum_{(k = 0)}^{t} R_k`}</Latex
    >
    and the return that is is going to be earned in the future
    <Latex>{String.raw`\sum_{k=t+1}^{H} R_k`}</Latex>.
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
    return. Just as with real life the action has no impact on the past. It
    turns out that ignoring past returns reduces the variance.
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
    make is to introduce discounting. Discounting accounts for the time value of
    rewards and additionally reduces the variance in policy gradients methods.
  </p>
  <div class="separator" />

  <h2>REINFORCE</h2>
  <p>
    The temporal decomposition and the introduction of discounting leads to an
    algorithm which is often called REINFORCE.
  </p>
  <div class="separator" />
</Container>
