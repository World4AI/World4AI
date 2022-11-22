<script>
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Policy Gradient Methods</title>
  <meta
    name="description"
    content="Policy gradient methods allow us to estimate and improve the policy directly without the need for a value function. Policy gradient methods are usually more stable than value based methods, but show hight variance."
  />
</svelte:head>

<h1>Policy Gradient Methods</h1>
<div class="separator" />

<Container>
  <p>
    The methods we have considered so far were designed to estimate the value
    function of a policy. The policy of the agent was determined implicitly by
    evaluating state-action pairs and taking the actions with the highest action
    value. In this chapter we are going to study methods that will allow us to
    learn the policy directly without using value functions. There are several
    reasons why we would prefer policy based methods over value based methods.
    Below is a subset of those reasons.
  </p>
  <p>
    Q-learning does not easily work with a large numer of actions or with
    continuous action spaces, because of the max operation that is required to
    determine the best action. If the action space is large or continuous the
    maximization operation becomes very involved and the algorithm becomes very
    inefficient. In policy gradient methods we sample an action from a
    probability distribution, thereby removing the need for a maximization
    operation.
  </p>
  <p>
    It is easy to implement a stochastic policy with policy gradient methods.
    This avoids the need for an additional exploration strategy, as we can
    randomly sample from the distribution and thereby explore the environment.
    Through gradient descent better actions are going to be assigned higher
    probabilities while bad actions will become unlikely.
  </p>
  <p>
    In Q-learning the action which constitutes the greedy action might change
    due to gradient descent. The change of the policy might be very abrupt and
    might thus destabilize training. In policy gradient methods we apply
    gradient descent to the policy directly, therefore the change of the
    probability distribution of actions is relatively smooth.
  </p>
  <p>
    The big disadvantage of policy gradient methods is the high variance, but
    adjustments to the naive implementation can be made to decrease the
    variance.
  </p>
  <div class="separator" />
</Container>
