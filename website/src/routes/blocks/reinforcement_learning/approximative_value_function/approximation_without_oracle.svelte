<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Value Approximation without an Oracle</h1>
<Question
  >How can we find a good value function approximation when the true value
  function is unknown?
</Question>
<div class="separator" />

<p>
  The assumption that there is some oracle that gives us the correct state value
  or action value is obviously not realistic. Therefore we have to interact with
  the environment to generate the approximate value function. As per usual the
  choice is between Monte-Carlo and TD learning.
</p>

<h2>Monte Carlo</h2>

<p>
  With Monte Carlo methods the state value and action value from the oracle is
  replaced by the return <Latex>G_t</Latex>.
</p>
<p>
  In Monte Carlo prediction the agent has to play a full episode before a single
  update to the weights can be done. After the episode finishes the weights are
  updated for each step taken during the episode. Instead of the true state
  value function the return is used as an approximation.
</p>
<Latex
  >{String.raw`
\begin {aligned}
  w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[G_{t}(S_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
  & = w_t + \alpha[G_{t}(S_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
\end {aligned}
`}</Latex
>

<p>
  In Monte Carlo control the agent tries to estimate the true action-value
  function using returns.
</p>
<Latex
  >{String.raw`
\begin {aligned}
  w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[G_t(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]^2 \\
  & = w_t + \alpha[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{q}(S_t, A_t, \mathbf{w}_t)
\end {aligned}
`}</Latex
>

<h2>Temporal Difference</h2>

<p>
  Temporal Difference methods use bootstrapped values as an approximation for
  the true state or action values.
</p>

<p>In TD prediction the state value function needs to be approximated.</p>
<Latex
  >{String.raw`
 \begin {aligned}
    w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[R_{t+1} + \hat{v}(S_{t+1},\mathbf{w}_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
    & = w_t + \alpha[R_{t+1} + \hat{v}(S_{t+1},\mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
 \end {aligned}
`}</Latex
>
<p>In TD control the action value function needs to be approximated.</p>
<Latex
  >{String.raw`
\begin {aligned}
  w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[R_{t+1} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]^2 \\
  & = w_t + \alpha[R_{t+1} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
\end {aligned}
  `}</Latex
>
<div class="separator" />
