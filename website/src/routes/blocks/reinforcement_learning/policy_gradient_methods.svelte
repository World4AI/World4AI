<script>
  import Question from "$lib/Question.svelte";
</script>

<h1>Policy Gradient Methods</h1>
<Question>What are policy gradient methods?</Question>
<div class="separator" />

<p>
  The methods we have considered so far were focused on estimating the value
  functions of a policy. The policy of the agent was determined implicitly by
  evaluating state-action pairs. In this chapter we are going to introduce
  methods that will allow us to learn the policy directly without using value
  functions.
</p>
<p>
  Before we move to the derivation of the policy gradient, let us discuss why it
  might be a good idea to use policy gradient methods. Below is a list of points
  that are often mentioned in the reinforcement learning literature and online
  lectures.
</p>
<p>
  Sometimes it is easier to estimate the policy directly, instead of estimating
  the action value function.
</p>
<p>
  Q-Learning does not easily allow the use of continuous action spaces, because
  of the max operation to determine the best action. In policy gradient methods
  we can sample an action from a continuous distribution.
</p>
<p>
  It is easy to implement a stochastic policy with policy gradient methods. This
  avoids the need for an additional exploration strategy, as we can randomly
  sample from the distribution. Through learning better actions are going to be
  assigned higher probabilities while bad actions will be unlikely.
</p>
<p>
  Policy gradient methods have better convergence properties. When in Q-Learning
  the action which constitutes the greedy action changes due to gradient
  descent, the change in the shape of the value function is abrupt and might
  destabilize training. In policy gradient methods the change of the probability
  distribution of actions is relatively smooth.
</p>
<p>
  Policy gradient methods have high variance, but improvements can be made to
  decrease the variance.
</p>
