<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Actor-Critic Methods</title>
  <meta
    name="description"
    content="Actor-Critic methods combine value based and policy based algorithms. The actor is the decision maker, the policy and the critique is the bootstrapped value function that evaluates the actions of the actor."
  />
</svelte:head>

<h1>Actor Critic Methods</h1>
<div class="separator" />

<Container>
  <p>
    So far we have studied value based methods like DQN, which estimate state or
    action value functions and determine the policy implicitly. We additionally
    derived policy based methods like REINFORCE, which estimate the policy
    directly. It turns out that combining both types of methods can result in
    so-called actor-critic methods. The actor is the decision maker and the
    policy <Latex>\pi</Latex>
    of the agent. The critic is the value function <Latex>V</Latex> that estimates
    how good or bad the decisions are that the actor makes. We will implement both
    functions as neural networks and train them simultaneously. The actor-critic
    methods can have significant improvements over pure value or policy based methods
    and in many cases constitute state of the art methods.
  </p>
  <p>
    At this point in time a natural question might occur. Is the REINFORCE
    algorithm with baseline (vpg) an actor-critic algorithm?
  </p>
  <Latex
    >{String.raw`
 \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\Big[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) [R(\tau)_{t:H} - V_w(S_t)]\Big] \\
`}</Latex
  >
  <p>
    REINFORCE with baseline has a policy <Latex
      >{String.raw`\pi_{\theta}`}</Latex
    >
    and a value function <Latex>{String.raw`V_w`}</Latex>. Should that be
    sufficient to classify the algorithm as actor-critic? It turns out that not
    all agents that have separate policy and value functions are defined as
    actor-critic methods. The key component that is required is bootstrapping.
  </p>
  <Latex
    >{String.raw`
 \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\Big[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) [R_{t+1} + V_w(S_{t+1}) - V_w(S_t)]\Big] 
`}</Latex
  >
  <p>
    The next state <Latex>{String.raw`S_{t + 1}`}</Latex> that results from the action
    of the actor has to be evaluated by the value function, the critic. When we calculate
    <Latex>{String.raw`V(S_{t + 1})`}</Latex> we ask the critic to calculate the
    expected value of the next state<Latex>{String.raw`S_{t+1}`}</Latex> that resulted
    from the actor taking an action <Latex>{String.raw`A_t`}</Latex>. That way
    the critic essentially "critiques" the previous action of the actor.
  </p>
  <p>
    In the following sections we will discuss some of the actor-critic
    algorithms. Especially we will focus on so called advantage actor-critic
    methods.
  </p>
  <div class="separator" />
</Container>
