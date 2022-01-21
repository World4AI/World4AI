<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";
  import Latex from "$lib/Latex.svelte";
  import Button from "$lib/Button.svelte";
  import Code from "$lib/Code.svelte";

  let distributions = [
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -100, max: 100, expectation: 0 },
    { min: -1, max: 2, expectation: 0.5 },
  ];

  let estimates = [];
  let estimates2 = [];
  let alpha = 0.001;

  for (let i = 0; i < distributions.length; i++) {
    estimates.push(0);
    estimates2.push(0);
  }

  function argMax(arr) {
    return arr.reduce((iMax, x, i, a) => (x > a[iMax] ? i : iMax), 0);
  }

  function drawSamples(num) {
    for (let i = 0; i < num; i++) {
      distributions.forEach((dist, idx) => {
        let target = Math.random() * (dist.max - dist.min) + dist.min;
        let target2 = Math.random() * (dist.max - dist.min) + dist.min;
        estimates[idx] = estimates[idx] + alpha * (target - estimates[idx]);
        estimates2[idx] = estimates2[idx] + alpha * (target2 - estimates2[idx]);
      });
    }
    createTables();
  }

  function drawSpecifiedSamples() {
    drawSamples(1000000);
  }

  let header = ["Min", "Max", "Expectation", "Estimation"];
  let header2 = ["Min", "Max", "Expectation", "Estimation 1", "Estimation 2"];
  let data = [];
  let data2 = [];

  function createTables() {
    data = [];
    data2 = [];
    distributions.forEach((dist, idx) => {
      let dataPoint = [];
      let dataPoint2 = [];
      dataPoint.push(dist.min);
      dataPoint.push(dist.max);
      dataPoint.push(dist.expectation);
      dataPoint.push(estimates[idx]);
      dataPoint2 = [...dataPoint, estimates2[idx]];
      data.push(dataPoint);
      data2.push(dataPoint2);
    });
  }
  $: maxIdx = argMax(estimates);
  drawSpecifiedSamples();
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Double Q-learning</title>
  <meta
    name="description"
    content="Some reinforcement learning algorithms suffer from the overestimation bias. Double Q-learning is a technique that reduces the bias by introducing double estimators for the action value function."
  />
</svelte:head>

<h1>Double Q-Learning</h1>
<Question
  >What is overestimation bias and how can double Q-learning help?</Question
>
<div class="separator" />

<p class="info">
  "In some stochastic environments the well-known reinforcement learning
  algorithm Q-learning performs very poorly. This poor performance is caused by
  large overestimations of action values. These overestimations result from a
  positive bias that is introduced because Q-learning uses the maximum action
  value as an approximation for the maximum expected action value"
</p>
<div class="separator" />
<p>
  Below we see the update rule that is used in Q-learning. The <Latex
    >\max</Latex
  > operator that is present in the update step is the cause of a problem called
  overestimation bias, which can degrade the performance of Q-learning and slow down
  convergence.
</p>
<Latex
  >{String.raw`Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]`}</Latex
>
<p>
  Below we see an interactive example, a bandit with 11 arms, where the rewards
  are uniformly distributed. The rewards for the first 10 actions are randomly
  distributed between values -100 and 100, which results in an expected value of
  0. The last action is uniformly randomly distributed between -1 and 2, which
  results in an expected value of 0.5. If we knew the distribution beforehand,
  we would always select the last action, but instead we need to use the average
  over the drawn samples. In the example we draw 1,000,000 samples and use an <Latex
    >\alpha</Latex
  > of 0.001 to calculate the estimate. Due to randomness some of the actions will
  exhibit an estimate above 0 while some will exhibit estimates below 0, even though
  the expected value is 0. It is likely that some other action is going to have a
  higher average than the last action, which will make us overestimate the action
  value, this is the overestimation bias.
</p>
<Table {header} {data} idxHighlight={maxIdx} />
<Button on:click={drawSpecifiedSamples} value={"Draw Samples"} />
<p>
  The solution is to create two estimators for each action that sample the
  rewards independent of each other. With the first estimator (estimator 1) we
  select the action with the max estimation, but we use the value from the
  second estimation (estimator 2) to calculate the expected value. When you use
  the playground below, most of the time estimation 2 is going to be far below
  estimation 1. Even though randomness is contained within both estimators, it
  is unlikely that the same action is going have the same level of
  overestimation in both sets of samples. This technique therefore reduces the
  oversestimation bias and generally imroves convergence.
</p>
<Table header={header2} data={data2} idxHighlight={maxIdx} />
<Button on:click={drawSpecifiedSamples} value={"Draw Samples"} />
<p>
  Double Q-learning applies this technique to deal with the overestimation bias
  in Q-learning. We utilize two independent action value functions: <Latex
    >Q_1</Latex
  > and <Latex>Q_2</Latex>. At each timestep we throw a fair coin to decide
  which of the two value functions should be updated and which should be used in
  the calculation of the target value. To select actions <Latex>a</Latex>
  during <Latex>\epsilon</Latex>-greedy policy we combine both action value
  functions, using either the average <Latex>(Q_1 + Q_2)/2</Latex> or the sum <Latex
    >Q_1 + Q_2</Latex
  >. In bootstrapping we utilize the action value function that is being updated
  to pick the action and we utilize the other action value function to calculate
  the bootstrapped target value. Below we see an example of the update step
  where the fair coin determined to update the first action value function <Latex
    >Q_1</Latex
  >.
</p>
<Latex
  >{String.raw`Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_aQ_1(S_{t+1}, a)) - Q_1(S_t, A_t)]`}</Latex
>
<p>
  Below is a code snippet showing how double Q-learning might be implemented
  using Python and NumPy.
</p>
<Code
  code={`
def q_learning(env, obs_space, action_space, num_episodes, alpha, gamma, epsilon):
    # q as action value function
    Q_a = np.zeros(shape=(len(obs_space), len(action_space)))
    Q_b = np.zeros(shape=(len(obs_space), len(action_space)))
                 
    # epsilon greedy policy
    def policy(obs):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            Q = (Q_a + Q_b)/2
            action = Q[obs].argmax()
        return action
    
    for episode in range(num_episodes):
        # reset variables
        done, obs = False, env.reset()
        
        while not done:
            action = policy(obs)
            next_obs, reward, done, _ = env.step(action)
            
            # decide which Q function to update and which to use for target
            if np.random.rand() < 0.5:
                Q_1 = Q_a
                Q_2 = Q_b
            else:
                Q_1 = Q_b
                Q_2 = Q_a

            next_action = Q_1[next_obs].argmax()
            Q_1[obs][action] = Q_1[obs][action] + alpha * (reward + gamma * Q_2[next_obs][next_action] * (not done) - Q_1[obs][action])
            obs = next_obs
    
    # greedy policy
    Q = (Q_a + Q_b)/2
    policy_mapping = np.argmax(Q, axis=1)
    policy = lambda x: policy_mapping[x]
        
    return policy, Q
    `}
/>
<div class="separator" />
