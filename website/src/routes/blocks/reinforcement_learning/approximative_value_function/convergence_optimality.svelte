<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";
  import Latex from "$lib/Latex.svelte";

  let predictionHeader = ["Algorithm", "Tabular", "Linear", "Non-Linear"];
  let predictionData = [
    [
      "Monte Carlo",
      "Convergence (True Value Function)",
      "Convergence (Global Optimum)",
      "Convergence (Local Optimum)",
    ],
    [
      "Sarsa",
      "Convergence (True Value Function)",
      "Convergence (Near Global Optimum)",
      "No Convergence",
    ],
    [
      "Q-Learning",
      "Convergence (True Value Function)",
      "No Convergence",
      "No Convergence",
    ],
  ];

  let controlHeader = ["Algorithm", "Tabular", "Linear", "Non-Linear"];
  let controlData = [
    [
      "Monte Carlo",
      "Convergence (True Value Function)",
      "Oscilates",
      "No Convergence",
    ],
    [
      "Sarsa",
      "Convergence (True Value Function)",
      "Oscilates",
      "No Convergence",
    ],
    [
      "Q-Learning",
      "Convergence (True Value Function)",
      "No Convergence",
      "No Convergence",
    ],
  ];
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Approximation Convergence and
    Optimality
  </title>
  <meta
    name="description"
    content="Non linear function approximators, especially in combination with off policy temporal difference learning exhibit very poor convergence properties."
  />
</svelte:head>

<h1>Convergence and Optimality</h1>
<Question
  >What do we mean when we talk about convergence and optimality?</Question
>
<div class="separator" />
<p>
  The algorithms that we discussed during the last chapters attempt to find
  weights that create an approximate function that is as close as possible to
  the true state or action value function. The measurement of closeness that is
  used throughout reinforcement learning is the mean squared error (MSE). But in
  what way does finding the weights that produce the minimal mean squared error
  contribute to a value function that is close to the optimal function and are
  we guaranteed to find such weights?
</p>

<h2>Convergence</h2>

<p>
  When we talk about convergence we usually mean that as time moves along, the
  value function of the agent changes towards some specific form. The steps
  towards that form get smaller and smaller and our function should have the
  desired form in the limit.
</p>

<p>
  What does convergence mean for the prediction problem? For tabular methods we
  aspire to find the true value function of a policy <Latex>\pi</Latex>.
  Therefore convergence means that the value function of the agent converges
  towards the true value function. For approximative methods the agent adjusts
  the weight vector through gradient descent to reduce the mean squared error
  and if convergence is possible the weights move towards a specific vector.
  That does not necessarily mean that the agent finds a weight vector that
  generates the smallest possible MSE, as gradient descent might get stuck in a
  local minimum.
</p>
<Table header={predictionHeader} data={predictionData} />
<p>
  Monte Carlo and TD algorithms converge towards the true value function of <Latex
    >\pi</Latex
  >
  when the agent deals with finite MDPs and uses tabular methods. When we talk about
  approximate solutions the answers to the question whether prediction algorithms
  converge depend strongly on the type of algorithm. Monte Carlo algorithms use returns
  as a proxy for the true value function. Returns are unbiased but noisy estimates
  of the true value function, therefore we have a guarantee of convergence while
  using gradient descent. Linear methods converge to global optimum while non-linear
  methods (like neural networks) converge to a local optimum. The MSE for linear
  monte carlo approximators is convex, which means that there is a single optimum
  which is guaranteed to be found. The MSE for non-linear monte carlo approximators
  is non-convex, therefore gradient descent might get stuck in a local optimum. Temporal
  difference methods use bootstrapping. These algorithms use estimates for the target
  values in the update step. That makes them biased estimators. Q-Learning especially
  is problematic as there is no convergence guarantee even for linear methods.
</p>

<p>
  What does convergence mean for control? For tabular methods that means to find
  the optimal value function and thereby policy. Therefore convergence means
  that the value function of the agent converges towards the optimal value
  function. For approximative methods convergence means that gradient descent
  finds either a local or a global optimum, while trying to find the weights
  that minimize the mean squared error between the approximate value function
  and the optimal value function.
</p>
<Table header={controlHeader} data={controlData} />
<p>
  Linear functions (MC and SARSA) oscillate around the near optimal value. For
  off-policy learning and non-linear methods no convergence guarantees exist.
</p>

<h2>Optimality</h2>

<p>
  Finding the true optimal value function is not possible with function
  approximators, because the state space is continuous or very large. How should
  we decide then which algorithms to use? The most important takeaway should be
  that when deciding between algorithms, convergence should not be the primary
  decision factor. If it was then linear approximators would be the first
  choice. In practice off-policy temporal difference algorithms are often the
  first choice, even though according to the table above there is no convergence
  guarantee. The truth of the matter is that in practice neural networks work
  well, provided we use some particular techniques to prevent divergence. We
  will learn more about those in the next chapters.
</p>

<div class="separator" />
