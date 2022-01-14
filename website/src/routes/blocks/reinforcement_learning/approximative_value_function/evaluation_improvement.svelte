<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Approximation Evaluation and
    Improvement</title
  >
  <meta
    name="description"
    content="Similar to tabular reinforcement learning, with approximative value based reinforcement learning the agent has to utilize evaluation and improvement."
  />
</svelte:head>
<h1>Evaluation and Improvement</h1>

<Question
  >How does generalized policy iteration work with approximative value
  functions?</Question
>
<div class="separator" />

<p>
  Similar to dynamic programming and tabular reinforcement learning we are going
  to apply generalized policy iteration (GPI). In the policy evaluation step we
  are going to look for a value function <Latex>{String.raw`\hat{v} `}</Latex> that
  is as close as possible to the true value function <Latex
    >{String.raw`v_{\pi}`}</Latex
  >. In the policy improvement step we are going to utilize <Latex
    >{String.raw`\hat{q} `}</Latex
  > in order to act greedily and improve our policy.
</p>

<div class="separator" />
<h2>GPI with an Oracle</h2>
<p>
  We assume that we have some policy <Latex>\pi</Latex> and that we are interested
  in the true value function of that particular policy. Finding the true value function
  is out of the question du to the continuous observation space, so we have to find
  an approximation.
</p>
<Latex>{String.raw`\hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)`}</Latex>

<p>
  To build the theory that is going to be used throughout the rest of the book
  it is convenient to start the discussion by assuming that we are in a
  supervised learning setting and that there is an oracle that tells us what the
  true state-value <Latex>{String.raw`v_{\pi}(s) `}</Latex> for the given policy
  <Latex>\pi</Latex> and state <Latex>s</Latex> is. Later the discussion can be extended
  to reinforcement learning settings where the agent interacts with the environment.
</p>
<p>
  In supervised learning the goal is to find a weight vector <Latex
    >{String.raw`\mathbf{w}`}</Latex
  > that reduces the error between the true value function <Latex>v(s)</Latex> and
  <Latex>{String.raw`\hat{v}(s, \mathbf{w})`}</Latex>. In reinforcement learning
  usually mean squared error (MSE) is used as the measure of error.
</p>
<Latex
  >{String.raw`MSE \doteq \mathbb{E_{\pi}}[(v_{\pi} - \hat{v}(s, \mathbf{w}))^2]`}</Latex
>

<p>
  If we find the weight vector <Latex>{String.raw`\mathbf{w} `}</Latex> that minimizes
  the above expression, then we found an approximation that is as close as possible
  to the true value function given by the oracle. The common approach to find such
  a vector is to use stochastic gradient descent with the following update rule.
</p>
<Latex
  >{String.raw`
 \begin {aligned}
   w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[v_{\pi}(S_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
   & = w_t + \alpha[v_{\pi}(S_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
 \end {aligned}
  `}</Latex
>

<p>
  Where the gradient <Latex
    >{String.raw`\nabla\hat{v}(S_t, \mathbf{w}_t)`}</Latex
  > is a vector that contains partial derivatives of the approximative value function
  with respect to individual weights.
</p>
<Latex
  >{String.raw`\large \nabla \hat{v}(s, \mathbf{w}) \doteq 
  \begin{bmatrix}
  \frac{\partial \hat{v}(\mathbf{w})}{\partial w_1} \\ 
  \frac{\partial \hat{v}(\mathbf{w})}{\partial w_2} \\
  \cdots \\ 
  \frac{\partial \hat{v}(\mathbf{w})}{\partial w_n}
  \end{bmatrix}`}</Latex
>
<p>
  To apply the same idea to policy improvement we utilize the action-value
  function and apply stochastic gradient descent.
</p>
<Latex
  >{String.raw`
\begin {aligned}
    w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]^2 \\
    & = w_t + \alpha[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{q}(S_t, A_t, \mathbf{w}_t)
\end {aligned}
  `}</Latex
>
<div class="separator" />

<h2>GPI without an Oracle</h2>
<p>
  The assumption that there is some oracle that gives us the correct state value
  or action value is obviously not realistic. Therefore we have to interact with
  the environment to generate the approximate value function. As per usual the
  choice is between Monte Carlo and TD learning.
</p>
<p>
  With Monte Carlo methods the state value and action value from the oracle is
  replaced by the return <Latex>G_t</Latex>.
</p>
<Latex
  >{String.raw`
\begin {aligned}
  w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[G_{t}(S_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
  & = w_t + \alpha[G_{t}(S_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
\end {aligned}
`}</Latex
>
<Latex
  >{String.raw`
\begin {aligned}
  w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[G_t(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]^2 \\
  & = w_t + \alpha[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{q}(S_t, A_t, \mathbf{w}_t)
\end {aligned}
`}</Latex
>
<p>
  Temporal Difference methods use bootstrapped values as an approximation for
  the true state or action values.
</p>
<Latex
  >{String.raw`
 \begin {aligned}
    w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[R_{t+1} + \hat{v}(S_{t+1},\mathbf{w}_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
    & = w_t + \alpha[R_{t+1} + \hat{v}(S_{t+1},\mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
 \end {aligned}
`}</Latex
>
<Latex
  >{String.raw`
\begin {aligned}
  w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[R_{t+1} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]^2 \\
  & = w_t + \alpha[R_{t+1} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
\end {aligned}
  `}</Latex
>
<div class="separator" />
