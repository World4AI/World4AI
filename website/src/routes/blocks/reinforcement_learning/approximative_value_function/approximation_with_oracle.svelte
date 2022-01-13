<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Approximation with an Oracle</h1>
<Question
  >How can we find a good value function approximation, when an oracle tells us
  the values of the true value function?</Question
>
<div class="separator" />

<h2>Generalized Policy Iteration</h2>

<p>
  Similar to dynamic programming the general idea when using approximative
  functions is to switch between policy evaluation and policy improvement.
</p>

<p>
  In the policy evaluation step we are going to look for a function <Latex
    >{String.raw`\hat{v} `}</Latex
  > that is as close as possible to the true value function <Latex
    >{String.raw`v_{\pi}`}</Latex
  >.
</p>

<p>
  In the policy improvement step we are going to utilize <Latex
    >{String.raw`\hat{q} `}</Latex
  > in order to act greedily and improve our policy.
</p>
<Latex>{String.raw`\hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)`}</Latex>
<Latex>{String.raw`\hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)`}</Latex>
<h3>Policy Evaluation</h3>
<p>
  Let us as always assume that we have some policy pi and are interested in the
  true value function of that particular policy. Finding the true value function
  is out of the question, so we have to deal with an approximation.
</p>
<Latex>{String.raw`\hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)`}</Latex>

<p>
  Generally it might be sufficient for us to find an approximative value
  function that is just good enough. In this chapter we are going to discuss
  what constitutes a “good” approximation and how we can find the weight vector
  <Latex>{String.raw`\mathbf{w}`}</Latex> for that “good” approximation .
</p>

<p>
  To build the theory that is going to be used throughout the rest of the book
  it is convenient to start the discussion by assuming that we are in a
  supervised learning setting and that there is an oracle who tells us what the
  true state-value <Latex>{String.raw`v_{\pi}(s) `}</Latex> for the given policy
  <Latex>\pi</Latex> and state <Latex>s</Latex> is. Later the discussion can be extended
  to reinforcement learning settings where the agent interacts with the environment.
</p>
<p>
  In supervised learning the goal is to find a weight vector w that produces a
  function that fits the training data as close as possible. That means that we
  want weights that reduce the difference between the true state-value and our
  approximation as much as possible. In reinforcement learning Mean Squared
  Error (MSE) is used to define the difference between the true value function
  and the approximate value function.
</p>
<Latex
  >{String.raw`MSE \doteq \mathbb{E_{\pi}}[(v_{\pi} - \hat{v}(s, \mathbf{w}))^2]`}</Latex
>

<p>
  If we find the weight vector <Latex>{String.raw`\mathbf{w} `}</Latex> that minimizes
  the above expression, then we found an approximation that is as close as possible
  to the true value function given by the oracle.
</p>

<p>
  The common approach to find such a vector is to use stochastic gradient
  descent. Stochastic gradient descent in a setting with an oracle would work as
  follows. The agent interacts with the environment using the policy <Latex
    >\pi</Latex
  >. For each of the observations the agent calculates the approximate value and
  compares the difference between the approximate value and the true value given
  by the oracle using the mean squared error. In the next step the agent
  calculates the gradients of MSE with respect to the weights of the value
  function. Using the gradient the agent reduces the MSE by adjusting the weight
  vector <Latex>{String.raw`\mathbf{w}.`}</Latex> Stochastic gradient descent means
  that the update of the weights is done at each single step.
</p>
<p>The update rule for the weight vector is as follows.</p>
<Latex
  >{String.raw`
 \begin {aligned}
   w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[v_{\pi}(S_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
   & = w_t + \alpha[v_{\pi}(S_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
 \end {aligned}
  `}</Latex
>

<p>
  The gradient <Latex>{String.raw`\nabla\hat{v}(S_t, \mathbf{w}_t)`}</Latex> is a
  vector that contains partial derivations of the approximative value function with
  respect to individual weights. We reduce the weights into the direction of the
  gradient.
</p>
<Latex
  >{String.raw`\nabla \hat{v}(s, \mathbf{w}) \doteq (\frac{\partial f(\mathbf{w})}{\partial w_1}, \frac{\partial f(\mathbf{w})}{\partial w_2}, ... , \frac{\partial f(\mathbf{w})}{\partial w_d})^T`}</Latex
>

<p>
  Linear functions and neural networks are differentiable, decision Trees are
  not differentiable functions. That means that for linear functions (and neural
  networks) it is easy to determine how to adjust the weight vector <Latex
    >{String.raw`\mathbf{w}`}</Latex
  >. From now on we are primarily going to focus on neural networks. To discuss
  some of the theoretical properties we will return to linear methods during the
  next few chapters.
</p>

<h3>Policy Improvement</h3>
<p>
  Policy improvement with function approximators utilizes the action-value
  function instead of a state-value function.
</p>
<Latex>{String.raw`\hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)`}</Latex>

<p>
  Once again we assume to have an oracle that provides the true action-value for
  a policy <Latex>\pi</Latex>, given the state and the action. At each time step
  the agent selects an action using <Latex>\epsilon</Latex>-greedy. Using the
  information from the oracle and the approximate estimation, the agent adjusts
  the weights of the function to get as close as possible to the true
  action-value function.
</p>
<Latex
  >{String.raw`
\begin {aligned}
    w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]^2 \\
    & = w_t + \alpha[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{q}(S_t, A_t, \mathbf{w}_t)
\end {aligned}
  `}</Latex
>
