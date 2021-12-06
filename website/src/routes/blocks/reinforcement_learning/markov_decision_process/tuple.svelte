<script>
  import Question from '$lib/Question.svelte';
  import Math from '$lib/Math.svelte';
  import Grid from '$lib/reinforcement_learning/grid_world/Grid.svelte';
  import { GridEnvironment } from '$lib/reinforcement_learning/grid_world/GridEnvironment';
  import { gridMap } from '$lib/reinforcement_learning/grid_world/maps';

  let env_1 = new GridEnvironment(gridMap);
  let env_2 = new GridEnvironment(gridMap);
</script>

<svelte:head>
    <title>World4AI | Reinforcement Learning | MDP As Tuple</title>
    <meta name="description" content="A Markov decision process is a tuple that contains the state space, the action space, the transition probabilities matrix, the reward function and the discount factor gamma.">
</svelte:head>

<h1>MDP as Tuple</h1>
<div class="separator"></div>
<p>The most formal definition of a Markov decison process deals with the individual components of the MDP. A Markov decision process can be defined as a tuple <Math latex={"(\\mathcal{S, A}, P, R, \\gamma)"} />, where each individual component of that tuple is a required component of a valid MDP. The definition of each component is going to be important in many subsequent sections, because those definitions are the basis of most mathematical proofs in reinforcement learning.</p> 
<p>In the following sections we will take a look at each of the contents of the tuple individually.</p>
<div class="separator"></div>

<h2><Math latex={'\\mathcal{S}'} />: States</h2>
<p>In a Markov decision process <Math latex={'\\mathcal{S}'} /> is the state space, that contains all possible states of the environment.</p>
<div class='flex-center'>
  <Grid env={env_1} showGridOnly={true} showStateSpace={true}/>
</div>
<p>In the example above for example we are dealing with a 5X5 grid world, where each state is represented by a row and column tuple: (row, column). Alltogether there are exactly 25 possible states, therefore our state space looks as follows: <Math latex={'\\mathcal{S}=[(0,0), (0, 1), (0, 2), ... , (4, 4)]'} />.</p>
<p class="info"><Math latex={'\\mathcal{S}'} /> is the set of all legal states.</p>
<div class="separator"></div>

<h2><Math latex={'\\mathcal{A}'} />: Actions</h2>
<p>In a Markov decision process <Math latex={'\\mathcal{A}'} /> is the action space, that contains all possible actions of the environment.</p>
<div class='flex-center'>
  <Grid env={env_2} showGridOnly={true} showActionSpace={true}/>
</div>
<p>In this simple grid world the agent has the option to move into four different directions: north, east, south and west. The same actions are represented in the environment by four different numbers: 0, 1, 2, 3. For that reason the action space in that particular grid world is <Math latex={'\\mathcal{A}=[0, 1, 2, 3]'} />. Even if in some states it is not possible to move into a particular direction, the state space is usually kept consistent across the whole state space.</p>
<p class="info"><Math latex={'\\mathcal{A}'} /> is the set of all legal actions.</p>
<div class="separator"></div>

<h2><Math latex={'P'} />: Transitions</h2>
<p class="info"><Math latex={"P"} /> is the transition model.</p>
<Math latex={"P(s' \\mid s, a) \\doteq Pr[S_{t+1}=s' \\mid S_t=s, A_t=a]"} />
<p>The transition model is the function  that calculates the probability of landing in some state <Math latex={"s'"} /> at timestep <Math latex={"t+1"} /> when at timestep <Math latex={"t"} /> the state corresponds to <Math latex={"s"} /> and the action taken by the agent is <Math latex={"a"} />.</p>
<div class="separator"></div>

<h2><Math latex={'R'} />: Rewards</h2>
<p class="info"><Math latex={'R'} /> is the reward model.</p>
<Math latex={"R(s,a) \\doteq \\mathbb{E}[R_{t+1} \\mid S_{t}=s, A_{t}=a]"} />
<p>The reward model is the function that calculates the expected value of the reward given state <Math latex={"s"} /> and action <Math latex={"a"} /> at time step <Math latex={"t"} />.</p>
<div class="separator"></div>

<h2>Gamma: Discounts</h2>
<p class="info"><Math latex={"\\gamma"} /> (gamma) is the discount factor, where <Math latex={`0 \\leq \\gamma \\leq 1`} />.</p>
<p>Consider the following example. You can get 1000$ now or 1000$ in 10 years. What would you choose? The answer is hopefully 1000$ now. The reason every rational agent would choose 1000$ is the time value of money or generally speaking the time value of rewards. In the case of dollars you could invest the money for 10 years and get an amount that is larger. Therefore there should be a compensation if the agent decides to delay his reward. The gamma or discount factor is used to adjust the value of rewards. Future rewards are considered of less value.</p>
<p>The value of rewards from the perspective of the agent at time step <Math latex={"t"} /> is as following:</p>
<p>The value of a reward received at timestep <Math latex={"t+1"} /> is <Math latex={"\\gamma^0 * R_{t+1}"} /></p>
<p>The value of a reward received at timestep <Math latex={"t+2"} /> is <Math latex={"\\gamma^1 * R_{t+2}"} /></p>
<p>The value of a reward received at timestep <Math latex={"t+3"} /> is <Math latex={"\\gamma^2 * R_{t+3}"} /></p>
<p class="info"><strong>Episodic</strong> tasks are tasks that have a natural ending. The last state in an episodic task is called a <em>terminal state</em>. The letter <Math latex={"T"} /> is used to mark the final time step.</p>
<p class="info"><strong>Continuing</strong> tasks are tasks that do not have a natural ending and may theoretically go on forever.</p>
<p>Mathematically speaking if you are dealing with episodic tasks, like the Frozen Lake environment,  then the discount factors are not strictly required. For continuing tasks a discount factor is required. The reason for that is the need for the agent to maximize the expected sum of future rewards. If the task is continuing then the sum of rewards might become infinite and the agent can not deal with that. If the value of gamma is between 0 and 1 then the sum becomes finite.</p>
<p>Usually the value of gamma is between 0.9 and 0.99.</p>

<p>Let’s assume a gamma with a value of 0.9.</p>

<p>At <Math latex={"t+3"} />: the discount factor is <Math latex={"\\gamma^2 = 0.81"} /></p>
<p>At <Math latex={"t+5"} />: the discount factor is <Math latex={"\\gamma^4 = 0.66"} /></p>
<p>At <Math latex={"t+11"} />: the discount factor is <Math latex={"\\gamma^{10} = 0.35"} /></p>
<p>At <Math latex={"t+21"} />: the discount factor is <Math latex={"\\gamma^{20} = 0.12"} /></p>
<p>At <Math latex={"t+51"} />: the discount factor is <Math latex={"\\gamma^{50} = 0.005"} /></p>
<p>The discount factor keeps approaching 0, which makes the value of rewards in the far future almost 0. That prevents an infinite sum of rewards.</p> 
<p class="info">Gamma <Math latex={"\\gamma"} /> is defined as a part of an MDP, but in practice it is treated as a hyperparameter of the agent. Theoretically gamma should be obvious from the environment. In practice there is no clear indiciation regarding how large <Math latex={"\\gamma"} /> should be. Tweaking the value might produce better results, because gamma can have an impact on how fast the algorithm converges and how stable the learning process is.</p>
<div class="separator"></div>
