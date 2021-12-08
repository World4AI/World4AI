<script>
  import Question from '$lib/Question.svelte';
  import Math from '$lib/Math.svelte';
  import Trajectory from '$lib/Trajectory.svelte';
  let trajectory = [
    {
      type: "R",
      subscript: 1,
    },
    {
      type: "R",
      subscript: 2,
    },
    {
      type: "R",
      subscript: 3,
    },
    {
      type: "R",
      subscript: 4,
    },
    {
      type: "R",
      subscript: 5,
    },
    {
      type: "R",
      subscript: 6,
    },
    {
      type: "R",
      subscript: 7,
    },
    {
      type: "R",
      subscript: '8',
    },
  ]  
  let trajectoryNumbers = [
    {
      type: "-1",
      subscript: 1,
    },
    {
      type: "-1",
      subscript: 2,
    },
    {
      type: "-1",
      subscript: 3,
    },
    {
      type: "-1",
      subscript: 4,
    },
    {
      type: "-1",
      subscript: 5,
    },
    {
      type: "-1",
      subscript: 6,
    },
    {
      type: "-1",
      subscript: 7,
    },
    {
      type: "10",
      subscript: '8',
    },
  ]  
</script>

<h1>MDP Solution</h1>
<Question>What does it mean for the agent to solve the MDP?</Question>
<div class="separator"></div>

<p>Once a particular MDP has been defined the next logical step would be to solve that MDP. In this chapter we are going to discuss what it actually means to solve the MDP.</p>
<div class="separator"></div>

<h2>Return</h2>
<p>In order to simplify notation and to introduce new necessary definitions we will first introduce the notion of a return <Math latex={'G'} />. A return is simply the sum of rewards starting from some timestep <Math latex={'t+1'} /> and going either to some terminal state <Math latex={'T'} /> (episodic tasks) or to infinity (continuing tasks). The letter <Math latex={'G'} /> stands for <em>Goal</em>, because the goal of the environment is encoded in the rewards.</p> 
<p>In episodic tasks the return is the sum of rewards in a single episode from time step <Math latex={'t'} /> to the terminal time step <Math latex={'T'} />.</p>
<Math latex={String.raw`G_t = R_{t+1} + R_{t+2} + … + R_T`} />
<p>In continuing tasks the return is the sum of rewards starting at time step t and going to possibly infinity.</p>
<Math latex={String.raw`G_t = R_{t+1} + R_{t+2} + R_{t+3} + …  = \sum_{k=0}^\infty{R_{k+t+1}}`} />
<p>In order to calculate the return of an episode we have to play through the sequence of states, actions and rewards all the way through from the initial state to the terminal state. This sequence of states, actions and rewards of an episode is called a <em>trajectory</em>.</p> 
<Trajectory />
<p>The calculation of the return <Math latex={'G_t'} /> only requires the knowledge of rewards from a trajectory. In the example below the episode goes from timestep <Math latex={'0'} /> until the terminal timestep <Math latex={'T = 8'} />. For each timestep taken the agent receives a negative reward of -1 and a reward of +10 when the agent reaches the terminal state <Math latex={'S_T'} />.</p>
<Trajectory {trajectory} />
<Trajectory trajectory={trajectoryNumbers} />
<p>Based on that trajectory the undiscounted returns from the perspective of time step 0, 1, and 2 look as follows.</p>
<Math latex={String.raw`
\begin{aligned}
   G_0 & = R_1 + R_2 + R_3 + R_4 + R_5 + R_6 + R_7 + R_8 \\
   & = (-1) + (-1) + (-1) + (-1) + (-1) + (-1) + (-1) + 10 = 3
\end{aligned}
`} />
<Math latex={String.raw`
\begin{aligned}
   G_1 & = R_2 + R_3 + R_4 + R_5 + R_6 + R_7 + R_8 \\
   & = (-1) + (-1) + (-1) + (-1) + (-1) + (-1) + 10 = 4
\end{aligned}
`} />
<Math latex={String.raw`
\begin{aligned}
   G_2 & = R_3 + R_4 + R_5 + R_6 + R_7 + R_8 \\
   & = (-1) + (-1) + (-1) + (-1) + (-1) + 10 = 5
\end{aligned}
`} />

<p>To avoid an infinite return (in continuing tasks), future rewards  are discounted by <Math latex={String.raw`\gamma`} />. Episodic tasks use discounting to emphasize the time value of rewards.</p> 
<Math latex={String.raw`G_t = R_{t+1} + \gamma{R_{t+2}} + \gamma^2{R_{t+3}} + …  = \sum_{k=0}^\infty{\gamma^k{R_{k+t+1}}}`}  />
<p>Looking at the same example from above the return <Math latex={String.raw`G_0`} /> looks as follows when we assume a gamma <Math latex={String.raw`\gamma`} /> of 0.9.</p>
<Math latex={String.raw`
   \begin{aligned} 
   G_0 & = R_1 + 0.9R_2 + 0.9^2 R_3 + 0.9^3 R_4 + 0.9^4 R_5 + 0.9^5 R_6 + 0.9^6 R_7 + 0.9^7 R_8 \\
   &= -1 + 0.9 * (-1) + 0.9^2 * (-1) + 0.9^3 * (-1) + 0.9^4 * (-1) + 0.9^5 * (-1) + 0.9^6 * (-1) + 0.9^7 * 10 \\ 
   \end{aligned}
`} />
<div class="separator"></div>

<h2>Policy</h2>
<p>If a policy is <em>deterministic</em> we define a policy as a mapping from state <Math latex={`s`} /> to action <Math latex={'a'} />. In that case the notation that we use for policy is <Math latex={String.raw`\mu(s)`} />. To generate an action <Math latex={'A_t'} /> at timestep <Math latex={'t'} /> we input the state <Math latex={'S_t'} /> into the policy function:  <Math latex={String.raw`A_t = \mu(S_t)`} />.</p>
<p>If a policy is <em>stochastic</em> we define a policy as a mapping from a state <Math latex={'s'} /> to a probability of an action <Math latex={'a'} /> and the mathematical notation is <Math latex={String.raw`\pi{(a \mid s)} = Pr[A_t = a \mid S_t = s]`} />. This notation can also be applied in a deterministic case. For a deterministic policy <Math latex={String.raw`\pi{(a \mid s) = 1}`} /> for for the selected action and <Math latex={String.raw`\pi{(a \mid s) = 0}`} /> for the rest of the legal actions. To generate an action we consider <Math latex={String.raw`\pi{(. \mid S_t)}`} /> to be the distribution of actions given states, where actions are draws from a policy distribution <Math latex={String.raw`A_t \sim \pi{(. \mid S_t)}` } />.</p>
<div class="separator"></div>

<h2>Value Functions</h2>
<p>Value functions map states or state-action pairs to “goodness” values, where goodness is expressed as the expected sum of rewards. Higher values mean more favorable states or state-action pairs.</p>
<p><em>State-Value Function:</em> <Math latex={String.raw`v_{\pi}(s) = \mathbb{E_{\pi}}[G_t \mid S_t = s]`} /></p>
<p><em>Action-Value Function:</em> <Math latex={String.raw`q_{\pi}(s, a) = \mathbb{E_{\pi}}[G_t \mid S_t = s, A_t = a]`} /></p>
<p>The state-value function expresses the expected return when following a particular policy <Math latex={String.raw`\pi`} /> given the state <Math latex={String.raw`s`} />. The action-value function expresses the expected return given the state <Math latex={`s`} /> while taking the action <Math latex={`a`} /> in the current step and following the policy <Math latex={String.raw`\pi`} /> afterwards.</p>  
<div class="separator"></div>

<h2>Optimality</h2>
<p>At the beginning of the chapter we asked ourselves what it means for an agent to solve a Markov decision process. The solution of the MDP means that the agent has learned an optimal policy function.</p> 
<p class="info">To solve the MDP is to find the optimal policy.</p>
<p>Optimality implies that there is a way to compare different policies and to determine which of the policies is better. Policies are evaluated in terms of their value functions</p>
<p><Math latex={String.raw`\pi \geq \pi’` } /> if and only if <Math latex={String.raw`v_{\pi}(s) \geq v_{\pi'}(s)` } /> for all <Math latex={String.raw`s \in \mathcal{S}`} /> </p>
<p>In finite MDPs value functions are used as a metric of the goodness of a policy. The policy  <Math latex={String.raw`\pi`} /> is said to be better than the policy <Math latex={String.raw`\pi’`} /> if and only if the value function of <Math latex={String.raw`\pi`} /> is larger or equal to the value function of policy <Math latex={String.raw`\pi’`} /> for all states in the state set S.</p> 
<p>The optimal policy <Math latex={String.raw`\pi_*`} /> is defined as:</p>
<p><Math latex={String.raw`\pi_* \geq \pi`} /> for all <Math latex={String.raw`\pi` } /></p>
<p>The optimal policy is the policy that is better (or at least not worse) than any other policy.</p>   
<p>The optimal state-value funtion:</p>
<p><Math latex={String.raw`v_*(s) = \max_{\pi} v_{\pi}(s)`} /> for all states <Math latex={String.raw`s \in \mathcal{S}`} /></p>
<p>The optimal action-value function:</p>
<p><Math latex={String.raw`q_*(s, a) = \max_{\pi} q_{\pi}(s, a)`} /> for all states <Math latex={String.raw`s \in \mathcal{S}` }/> and all actions <Math latex={String.raw`a \in \mathcal{A}`} /> </p>
<p>The state-value function and the action-value function that are based on the optimal policy are called optimal state-value and optimal action-value function respectively.</p> 
<p class="info">There might be several optimal policies, but there is always only one optimal value function.</p>
<div class="separator"></div>

<h2>Bellman Equations</h2>
<p>An important property of returns is that they can be expressed in terms of future returns.</p>
<Math latex={String.raw`
      \begin{aligned}
      G_t & = R_{t+1} + \gamma{R_{t+2}} + \gamma^2{R_{t+3}} + … \\
      & = R_{t+1} + \gamma{(R_{t+2} + \gamma{R_{t+3}} + ...)} \\
      & = R_{t+1} + \gamma{G_{t+1}}
      \end{aligned}
`} />

<p>By using the properties of returns <Math latex={`G_t`} /> where each return can be expressed in terms of future returns <Math latex={String.raw`G_t = r_{t+1} + \gamma G_{t+1}`} /> we can arrive at recursive equations, where a value of a state can be defined in terms of values of the next state.</p> 
<p>Bellman equation for the state-value function</p>
<Math latex={String.raw`

      \begin{aligned}
      v_{\pi}(s) & = \mathbb{E_{\pi}}[G_t \mid S_t = s] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]
      \end{aligned}
`} />
<p>Bellman equation for the action-value function</p>
<Math latex={String.raw`
      \begin{aligned}
      q_{\pi}(s, a) & = \mathbb{E_{\pi}}[G_t \mid S_t = s, A_t = a] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\
      & = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = a]
      \end{aligned}
`} />
<p>Equations of the above form are called Bellman equations, named after the mathematician Richard E. Bellman. At the very first glance it might not seem like the equations add additional benefit to the definition of value functions, but the recursive relationships is what makes many of the reinforcement learning algorithms work.</p> 
<p>Bellman Optimality Equation for the state-value function:</p>
<Math latex={String.raw`
      \begin{aligned}
      v_*(s) & = \max_{a} q_{{\pi}_*}(s, a) \\
      & = \max_{a} \mathbb{E_{\pi_{*}}}[G_t \mid S_t = s, A_t = a] \\
      & = \max_{a} \mathbb{E_{\pi_{*}}}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\
      & = \max_{a} \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]
      \end{aligned}
`} />
<p>Bellman Optimality Equation for the state-value function:</p>
<Math latex={String.raw`
      \begin{aligned}
      q_*(s, a) & = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] \\
      & = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a]
      \end{aligned}
`} />
<div class="separator"></div>
