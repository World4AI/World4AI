<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Generalized Advantage Estimation</title
  >
  <meta
    name="description"
    content="Generalized advantage estimation is a robust method to calculate the advantage, which allows to reduce variance whiel maintaining a sound level of the bias."
  />
</svelte:head>

<h1>Generalized Advantage Estimation</h1>
<div class="separator" />

<Container>
  <p class="info">
    We propose a family of policy gradient estimators that significantly reduce
    variance while main- taining a tolerable level of bias.
  </p>
  <p>We can rewrite the policy gradient in the following form.</p>
  <Latex
    >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t)  \Psi_t] 
    \end{aligned}
  `}</Latex
  >
  <p>
    The greek letter psi <Latex>\Psi</Latex> can be replaced by a variety of options,
    but in modern reinforcement learning algorithms it is most likely to contain
    an advantage estimation <Latex>A(S_t, A_t)</Latex>, making it an advantage
    actor-critic algorithm.
  </p>
  <Latex
    >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t)  A(S_t, A_t)] 
    \end{aligned}
`}</Latex
  >
  <p>
    Additionally we can define the number of steps <Latex>n</Latex>, where <Latex
      >n</Latex
    > defines how many returns have to be unrolled following a policy <Latex
      >{String.raw`\pi_{\theta}`}</Latex
    > before an optimization step is taken. When we want to distinguish the advantage
    function based on the number of steps, we define the advantage function as <Latex
      >{String.raw`A^{n}_t(S_t,A_t)`}</Latex
    >.
  </p>
  <Latex
    >{String.raw`
\begin{aligned}
	& \hat{A}^{(1)}_t(S_t, A_t) = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \\
	& \hat{A}^{(2)}_t(S_t, A_t) = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2}) - V(S_t) \\
    & \cdots \\
    & \cdots \\
    & \hat{A}^{(n)}_t(S_t, A_t) = \sum_{t'=t}^{t+n-1} \gamma^{t'-t}R_{t'+1} + \gamma^{t+n}V(S_{t+n}) - V(S_t) \\
\end{aligned}
  `}</Latex
  >
  <p>
    The higher the number n the higher the variance and the lower the bias. The
    sweetspot is usually not at the extreme ends, where n is either 1 and we end
    up with a one step temporal difference advantage estimation or n is
    unbounded and we end up with a full monte carlo estimation. In the A3C
    algorithm n corresponded to 5, but it is not clear if it was the right
    choice.
  </p>
  <p>
    The generalized advantage estimation allows us to utilize a mixture of many
    advantage functions <Latex>{String.raw`A^{n}_t(S_t, A_t)`}</Latex> with different
    <Latex>n</Latex>, to hopefully end up with a more robust advantage
    estimation. To come up with a better advantage estimator, each of the <Latex
      >n</Latex
    > estimators is weighted and summed up.
  </p>
  <Latex
    >{String.raw`
\hat{A}^{GAE}_t = \sum_{n=1}^\infty w_n \hat{A}_t^{(n)}
  `}</Latex
  >
  <p>
    To control the strenghts of the weight decay, the authors utilize
    <Latex>\lambda</Latex> and create an exponentially-weighted estimator.
  </p>
  <Latex
    >{String.raw`
    \hat{A}^{GAE}_t = (1 - \lambda)(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + ...)
  `}</Latex
  >
  <p>This estimator can be unpacked and reduced to the following form.</p>
  <Latex
    >{String.raw`
\hat{A}^{GAE}_t = \sum^\infty_{l=0}(\gamma\lambda)^l\delta_{t+l}
  `}</Latex
  >
  <p>
    Where <Latex>\delta_t</Latex> is the temporal difference error at the timestep
    <Latex>t</Latex>.
  </p>
  <Latex
    >{String.raw`
\begin{aligned}
  \delta_t & = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \\
  \delta_{t+1} & = R_{t+2} + \gamma V(S_{t+2}) - V(S_{t+1})
\end{aligned}
  `}</Latex
  >
  <div class="separator" />
</Container>
