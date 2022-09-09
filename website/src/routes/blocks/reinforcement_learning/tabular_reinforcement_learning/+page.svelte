<script>
  import Container from "$lib/Container.svelte";
  import Table from "$lib/Table.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

  let header = ["", "A_0", "A_1"];
  let data = [
    ["S_0", "0", "2"],
    ["S_1", "1", "3"],
    ["S_2", "2", "4"],
  ];
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Tabular Reinforcement Learning</title
  >
  <meta
    name="description"
    content="Tabular reinforcement learning deals with finding optimal value functions and policies for finite markov decision processes."
  />
</svelte:head>

<h1>Tabular Reinforcement Learning</h1>
<div class="separator" />

<Container>
  <p>
    Dynamic programming offers us valuable theoretical foundations, but
    generally speaking we do not have access to the model of the environment.
    The only viable solution that remains is learning through interaction with
    the environment. This chapter is dedicated to reinforcement learning in
    finite MDPs. Finite Markov decision processes have a finite state and action
    space. This allows for the implementation of state-value and action-value
    functions that can be stored in lists or tables. Those implementations are
    appropriately called tabular methods.
  </p>
  <Table {header} {data} />
  <p>
    Above we can see a so called <strong>Q-table</strong> for an imaginary gridworld
    with 3 states and 2 actions. The available actions are used as the column names,
    the available states are used as row names and the intersection contains the
    estimated Q-value for that specific state-action pair.
  </p>
  <p>
    To understand why Q-tables are essential tools that can help us find the
    optimal policy, let us remind ourselves how the value iteration algorithm
    works.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
    \text{(1: Policy Evaluation) } & q_{k+1}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')] \\
    \text{(2: Policy Improvement) }& v_{k+1}(s) = \max_a q_{k+1}(s, a)
  \end{aligned}
`}
  </Latex>
  <p>
    Value iteration consists of policy evaluation (that requires the knowledge
    of the model) and policy improvement (that takes the max over the
    action-value function). Ignoring the first step for now, once we have access
    to the action-value function (a Q-table in our case), all we really need to
    do in each state is to choose the action with the maximum value. Different
    tabular reinforcement learning algorithms utilize different methods to
    estimate the Q-values, but the policy improvement steps remain similar.
  </p>
  <p>
    Before we move on to the discussion of tabular reinforcement learning
    algorithms it is important to discuss the difference between on-policy and
    off-policy methods. We could ask ourselves: “Do we need to improve the same
    policy that is used to generate actions or can we learn the optimal policy
    while using the data that was produced by a different policy?”. To frame the
    question differently “Is it possible to learn the optimal policy while only
    selecting random actions?”. That depends on the design of the algorithm.
    On-policy methods improve the same policy that is also used to generate the
    actions, while off-policy methods improve a policy that is not the one that
    is used to generate the trajectories. We will encounter and implement both
    types of algorithms.
  </p>
  <p>
    The current state of the art reinforcement learning rarely deals with
    tabular methods any more, but it is still more convenient to start the
    exploration of reinforcement learning techniques with those, as the general
    ideas are extremely relevant to modern (approximative) methods which are
    going to be introduced in future sections.
  </p>
  <div class="separator" />
</Container>
