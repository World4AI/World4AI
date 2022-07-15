<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Policy Gradient Derivation</title>
  <meta
    name="description"
    content="The derivation of the policy gradient algorithm requires a succession of steps to arrive at a state where we do not require the knowledge of the MDP to calculate the gradient with respect to the weights."
  />
</svelte:head>

<h1>Policy Gradient Derivation</h1>
<div class="separator" />

<Container>
  <p>
    This derivation of the policy gradient theorem is based in the Deep RL
    Bootcamp lecture “Policy Gradients and Actor Critic” by Pieter Abbeel. It is
    more intuitive that any other derivation we have encountered and is
    therefore suited for beginners.
  </p>
  <p>
    In policy gradient methods in order to interact with the environment the
    agent utilizes a parametrized policy <Latex
      >{String.raw`\pi_{\theta}(a|s)`}</Latex
    >
    and no value function is involved. The policy provides a distribution of actions
    given the current state <Latex>s</Latex> and a set of learnable parameters <Latex
      >\theta</Latex
    >. In our case the policy is going to be a neural network with weights <Latex
      >\theta</Latex
    >. The neural network will generate a probability distribution and the agent
    will sample actions according to that distribution, <Latex
      >{String.raw`a \sim \pi_{\theta}(. \mid s)`}</Latex
    >.
  </p>

  <p>
    The interaction between the agent and the environment generates trajectories <Latex
      >\tau</Latex
    >, a sequence of tuples consisting of states, actions and rewards <Latex
      >{String.raw`(s_t, a_t, r_t, s_{t + 1}, a_{t + 1}, r_{t + 1}, ... , s_T, a_T, r_T)`}</Latex
    >. Each of the trajectories has a corresponding return <Latex>G</Latex>.
    When we talk about policy gradients, the return is defined as <Latex
      >R(\tau)</Latex
    > to indicate that the return is depends on the trajectory that was unrolled
    throught the interaction.
  </p>

  <p>
    The general goal of a reinforcement learning agent is to maximize the
    expected return. In policy gradient methods the expected return is also
    called the <Highlight>objective function</Highlight>
    <Latex>J</Latex>.
  </p>

  <Latex
    >{String.raw`
J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)] = \sum_{\tau}\mathbb{P}(\tau \mid \theta) R(\tau)
  `}</Latex
  >

  <p>
    The expectation is defined over the trajectories <Latex>\tau</Latex>, where
    each of the returns of a trajectory is weighted with the corresponding
    probability of that trajectory. For us this means that we have to find
    parameters
    <Latex>\theta</Latex> that generate trajectories with the highest expected return.
  </p>

  <Latex
    >{String.raw`
    \arg\max_{\theta}J(\pi_{\theta})
  `}</Latex
  >
  <p>For that purpuse we are going to use gradient ascent.</p>
  <Latex
    >{String.raw`
\theta \leftarrow \theta + \alpha \nabla_{\theta}J(\pi_{\theta})
  `}</Latex
  >
  <p>We can calculate the gradient using the following steps.</p>
  <Latex
    >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)] \\ 
    & = \nabla_{\theta} \sum_{\tau}\mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \sum_{\tau}\nabla_{\theta} \mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \sum_{\tau} \frac{\mathbb{P}(\tau \mid \theta)}{\mathbb{P}(\tau \mid \theta)} \nabla_{\theta} \mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \sum_{\tau} \mathbb{P}(\tau \mid \theta) \frac{\nabla_{\theta} \mathbb{P}(\tau \mid \theta)}{\mathbb{P}(\tau \mid \theta)} R(\tau) \\
    & = \sum_{\tau} \mathbb{P}(\tau \mid \theta) \nabla_{\theta} \log\mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\nabla_{\theta} \log\mathbb{P}(\tau \mid \theta) R(\tau)]
    \end{aligned}
  `}</Latex
  >
  <p>In the above reformulation we used a couple of mathematical tricks.</p>
  <p>
    First, from basic calculus we know that the derivative of a sum is a sum of
    derivatives. That allows us to bring in the derivative sign inside.
  </p>
  <p>
    Second, multiplying and dividing by the same number does not change the
    derivative calculation, because both operations cancel each other.
  </p>
  <p>
    Third, we use the likelihood ratio trick to rewrite part of the derivative
    as a log expression. The log has some nice properties that we are going to
    apply in a later step. The likelihood ratio trick utilizes the following
    identity.
  </p>
  <Latex
    >{String.raw`
\nabla_x \log f(\mathbf{x}) = \nabla_x f(\mathbf{x}) \frac{1}{f(\mathbf{x})}
  `}</Latex
  >
  <p>
    At this point in time we still face a problem, because we do not have access
    to the model of the environment and therefore do not know the derivative of
    <Latex>{String.raw`\mathbb{P}(\tau \mid \theta)`}</Latex>
    with respect to <Latex>\theta</Latex>. We need to reformulate the definition
    of the probability of the trajectory <Latex
      >{String.raw`\mathbb{P}(\tau \mid \theta)`}</Latex
    > in such a way that allows us to calculate the gradient.
  </p>
  <Latex
    >{String.raw`
\mathbb{P}(\tau \mid \theta) = \prod_t^H P(S_{t+1} \mid S_t, A_t) \pi_{\theta}(A_t \mid S_t)
  `}</Latex
  >
  <p>
    The probability of a trajectory <Latex>\tau</Latex> depends on one side on the
    policy of the agent <Latex>{String.raw`\pi_{\theta}`}</Latex>
    and on the other hand the model of the Markov decision process <Latex
      >P</Latex
    >. The interaction between the agent and the environment continues until the
    end of the episode, which is indicated by the horizon <Latex>H</Latex>. The
    probability of the full trajectory <Latex>\tau</Latex> is the product of individual
    probabilities of actions and states that occur within that trajectory. Currently
    we are dealing with the gradient of the log of the probability of a trajectory<Latex
      >{String.raw`\nabla_{\theta} \log \mathbb{P}(\tau \mid \theta)`}</Latex
    >, therefore we can rewrite the gradient in the following way.
  </p>

  <Latex
    >{String.raw`
    \begin{aligned}
    \nabla_{\theta} \log \mathbb{P}(\tau \mid \theta) & = \nabla_{\theta} \log (\prod_t^H P(S_{t+1} \mid S_t, A_t) \pi_{\theta}(A_t \mid S_t)) \\
    & = \nabla_{\theta} (\sum_t^H \log P(S_{t+1} \mid S_t, A_t) + \sum_t^H \log \pi_{\theta}(A_t \mid S_t)) \\
    & = (\sum_t^H \nabla_{\theta} \log P(S_{t+1} \mid S_t, A_t) + \sum_t^H \nabla_{\theta}  \log \pi_{\theta}(A_t \mid S_t)) \\
    & = \sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) \\
    \end{aligned}
  `}</Latex
  >
  <p>
    First, we realize that the log of a product is the sum of the logs, <Latex
      >{String.raw`\log(x*y) = \log x + \log y`}</Latex
    >. This makes obvious why the reformulation of the problem in terms of logs
    was a necessary step. This allows us to separate the policy from the model
    in a powerful way.
  </p>

  <p>
    Finally we realize that <Latex
      >{String.raw`\nabla_{\theta} \log P(S_{t+1} \mid S_t, A_t)`}</Latex
    > is 0. The derivative is with respect to <Latex>\theta</Latex>, which is
    the parameter vector of the policy and the policy has no impact on the
    model. No matter how the policy looks like, the agent can not change the
    underlying dynamics of the MDP.
  </p>
  <p>The final reformulation of the gradient looks as follows.</p>
  <Latex
    >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\nabla_{\theta} \log\mathbb{P}(\tau \mid \theta) R(\tau)] \\
    & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)]
    \end{aligned}
  `}</Latex
  >

  <p>
    The final gradient depends only on the gradient of the policy <Latex
      >\pi</Latex
    > and the realized return, the knowledge of the dynamics of the model are not
    required.
  </p>

  <p>
    The gradient is inside the expectation, therefore we can sample trajectories
    and estimate the true gradient. The larger the sample size the better the
    estimate. After we sample several trajetories <Latex>m</Latex> we take a gradient
    ascent step and repeat the process again.
  </p>
  <Latex
    >{String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) R(\tau^{(i)})
    \end{aligned}
  `}</Latex
  >
  <p>
    The above formulation can be put into a valid algorithm, but the
    implementation would be of high variance. Therefore we will first study some
    methods which will allow us to create an algorithm with lower variance.
  </p>
  <div class="separator" />
</Container>
