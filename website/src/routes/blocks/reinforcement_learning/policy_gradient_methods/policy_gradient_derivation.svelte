<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<h1>Policy Gradient Derivation</h1>
<Question>How can we derive the policy gradient method?</Question>
<div class="separator" />

<p>
  There are several different methods to derive and explain the policy gradient.
  The approach we are going to cover is based in the Deep RL Bootcamp lecture on
  “Policy Gradients and Actor Critic” by Pieter Abbeel. We find that this
  derivation more intuitive that any other we have encountered and therefore .
</p>
<p>
  In policy gradient methods in order to interact with the environment the agent
  utilizes a parametrized policy <Latex>{String.raw`\pi_{\theta}(a|s)`}</Latex>,
  instead of a value function. The policy provides a distribution of actions
  that is based on the current state and the learnable parameters <Latex
    >\theta</Latex
  >.
</p>

<p>
  In our case the policy is going to be a neural network with weights <Latex
    >\theta</Latex
  >. The neural network will generate a probability distribution that will be
  sampled to generate an action, <Latex
    >{String.raw`a \sim \pi_{\theta}(. \mid s)`}</Latex
  >.
</p>

<p>
  The interaction generates trajectories <Latex>\tau</Latex>, a sequence of
  tuples consisting of states, actions and rewards <Latex
    >{String.raw`(s_t, a_t, r_t, s_{t + 1}, a_{t + 1}, r_{t + 1}, ... , s_T, a_T, r_T)`}</Latex
  >. Each of the trajectories has a corresponding return <Latex>G</Latex>.
  Sometimes, especially when talking about policy gradients, the return is also
  defined as <Latex>R(\tau)</Latex> to indicate that the return is based on the trajectory
  that was followed.
</p>

<p>
  The general goal of a reinforcement learning agent is to maximize the expected
  sum of rewards. In policy gradient methods the expected return is also called
  the <Highlight>objective function</Highlight>.
</p>

<Latex
  >{String.raw`
J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)] = \sum_{\tau}\mathbb{P}(\tau \mid \theta) R(\tau)
  `}</Latex
>

<p>
  The expectation is defined over the trajectories <Latex>\tau</Latex> that are sampled
  using a policy <Latex>\pi</Latex> with parameters <Latex>\theta</Latex>. Each
  return that results from the trajectory <Latex>\tau</Latex> is weighted with the
  corresponding probability. For us this means that we have to find parameters
  <Latex>\tau</Latex> that generate trajectories with the highest expected returns.
</p>

<Latex
  >{String.raw`
    \arg\max_{\theta}J(\theta)
  `}</Latex
>
<p>
  To find the parameters that maximize the objective function we are going to
  use gradient ascent.
</p>
<Latex
  >{String.raw`
\theta \leftarrow \theta + \alpha \nabla_{\theta}J(\pi_{\theta})
  `}</Latex
>

<p>
  The objective function <Latex>{String.raw`J(\pi_{\theta})`}</Latex> is unknown,
  because the calculation of the expectation over trajectories would require the
  knowledge of the dynamics of the model. Therefore it is not that simple to calculate
  the gradient of the objective function. We need to restate the problem from a different
  perspective in order to calculate the gradient <Latex
    >{String.raw`\nabla_{\theta}`}</Latex
  >.
</p>

<p>The likelihood ratio trick utilizes the following identity.</p>
<Latex
  >{String.raw`
\nabla_x \log f(\mathbf{x}) = \nabla_x f(\mathbf{x}) \frac{1}{f(\mathbf{x})}
  `}</Latex
>
<p>
  Here we use the chain rule and the derivative of the log function to calculate
  the derivative.
</p>
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
  derivatives. That allows us to bring in the derivative sign inside. We will
  talk about the huge importance of that step down below.
</p>
<p>
  Second, multiplying and dividing by the same number does not change the
  derivative calculation, because both operations cancel each other. We multiply
  and divide by the probability of trajectory. Combining the sum over
  trajectories and the weighting with the probabilities of trajectories gives us
  an expectation over trajectories.
</p>
<p>
  Third, we use the likelihood ratio trick to rewrite part of the derivative as
  a log expression. The log has some nice properties that we are going to apply
  in a later step.
</p>

<p>
  At this point in time we still do not know the derivative of <Latex
    >{String.raw`\mathbb{P}(\tau \mid \theta)`}</Latex
  > with respect to <Latex>\theta</Latex>, because we do not know the exact
  model of the model of the MDP. We reformulate the probability <Latex
    >{String.raw`\mathbb{P}(\tau
      \mid \theta)`}</Latex
  >.
</p>
<Latex
  >{String.raw`
\mathbb{P}(\tau \mid \theta) = \prod_t^H P(S_{t+1} \mid S_t, A_t) \pi_{\theta}(A_t \mid S_t)
  `}</Latex
>

<p>
  The probability of a trajectory depends on one side on the policy of the agent <Latex
    >{String.raw`\pi_{\theta}`}</Latex
  >, which determines the probability of the action <Latex>a_t</Latex> based on the
  current state <Latex>s_t</Latex>. On the other hand the model calculates the
  probability of the next state <Latex>{String.raw`s_{t + 1}`}</Latex> based on the
  action taken
  <Latex>a_t</Latex> and the current state <Latex>s_t</Latex>. The selection of
  actions and next states continues until the end of the episode, which is
  indicated by the horizon <Latex>H</Latex> The calculation of the probability of
  the full trajectory is the product of individual probabilities that are calculated
  throughout the trajectory.
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
  It turns out that the gradient of the reformulated problem is an easier
  problem.
</p>

<p>
  First, we realize that the log of a product is the sum of the logs, <Latex
    >{String.raw`\log(x*y) = \log x + \log y`}</Latex
  >. This makes obvious why the reformulation of the problem in terms of logs
  was a necessary step. This allows us to separate the policy from the model in
  a powerful way.
</p>

<p>
  Finally we realize that <Latex
    >{String.raw`\nabla_{\theta} \log P(S_{t+1} \mid S_t, A_t)`}</Latex
  > is 0. The derivative is with respect to <Latex>\theta</Latex>, which is the
  parameter vector of the policy and the policy has no impact on the model. No
  matter how the policy looks like, the agent can not change the underlying
  dynamics of the MDP.
</p>
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
  > and the realized return, the knowledge of the dynamics of the model are not required.
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
  Let us also discuss why it was important to push the gradient inside the
  expectation. The gradient of the expectation <Latex
    >{String.raw`\nabla_{\theta}\mathbb{E}`}</Latex
  > implies that we have to know the expected value of returns to calculate the gradient,
  which we don’t. When the expectation is inside we can sample trajectories and estimate
  the true gradient. The larger the sample size the better the estimate. In practice
  often the gradient step is taken after a single episode, indicating <Latex
    >m = 1</Latex
  >
</p>

<p>
  We are not going to implement this naive policy gradient algorithm, as there
  is high variance due to high noise of returns of individual episodes. Starting
  with the next chapter we will investigate methods to decrease the variance and
  implement the algorithm in PyTorch.
</p>
