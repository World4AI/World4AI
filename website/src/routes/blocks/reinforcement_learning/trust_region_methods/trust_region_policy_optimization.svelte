<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Trust Region Policy Optimization (TRPO)</h1>
<Question
  >How does TRPO solve the problems that are inherent to policy gradient
  methods?</Question
>
<div class="separator" />

<p>
  We will start our discussion of Trust Region Policy Optimization (TRPO) with a
  slight reformulation of our objective.
</p>
<Latex
  >{String.raw`
    J(\theta) = \mathbb{E}_{\pi_{\theta}}\Big[\sum_t \gamma^t R_t\Big]
  `}</Latex
>
<p>
  The objective we are usually trying to maximize is the expected sum of
  discounted rewards, as defined above. Now imagine we have two policies at our
  disposal : the new policy <Latex>J(\theta')</Latex> that we are trying to optimize
  and some old policy <Latex>J(\theta)</Latex>. We can define our objective as
  the performance difference between the two policies.
</p>
<Latex
  >{String.raw`
    J(\theta') - J(\theta) 
  `}</Latex
>
<p>
  Maximizing the distance between the two performance measures with respect to <Latex
    >\theta'</Latex
  > is the same as optimizing <Latex>J(\theta')</Latex>. That is because <Latex
    >J(\theta)</Latex
  > is essentially a constant when we calculate a gradient with respect to <Latex
    >\theta'</Latex
  >.
</p>
<p>
  The actual distance between the two performance measure is <Latex
    >{String.raw`\mathbb{E}_{\pi_{\theta'}}\Big[\sum_t \gamma^t A^{\theta}\Big]`}</Latex
  >, where we use the advantage function calculated based on the old policy with
  parameters <Latex>\theta</Latex>, but the expectation is calcualated based on
  the new policy with parameters <Latex>\theta'</Latex>. When we chage a policy
  from <Latex>{String.raw`\pi_{\theta}`}</Latex> to <Latex
    >{String.raw`\pi_{\theta'}`}</Latex
  > we change the trajectories that the policy generates. Each of the new states
  and actions that form the trajectory produce advantages compared to the old policy
  and the sum of those advantages is the difference in policies. If we look closely
  we can additionally recognize, that as long as the advantages are positive the
  policy <Latex>{String.raw`\pi_{\theta'}`}</Latex> is going to be an improvement
  over the policy <Latex>{String.raw`\pi_{\theta}`}</Latex>.
</p>
<Latex
  >{String.raw`
    J(\theta') - J(\theta) = \mathbb{E}_{\pi_{\theta'}}\Big[\sum_t \gamma^t A^{\theta}\Big]
  `}</Latex
>
<p>
  We can decompose the expectation above into two parts. The part that is
  dependent on the dynamics of the Markov Decision Process and the part that is
  dependent on the policy. The dynamics <Latex
    >{String.raw`\rho_{\pi_{\theta'}}(s)`}</Latex
  > tell us how often in expectation does the agent land in state <Latex
    >s</Latex
  >. The policy <Latex>{String.raw`\pi_{\theta'}`}</Latex> is the mapping from states
  to actions according to the new policy.
</p>
<Latex
  >{String.raw`
    J(\theta') - J(\theta) = \sum_s \rho_{\pi_{\theta'}}(s) \sum_a \pi_{\theta'}(a) \Big[A^{\theta}(s, a)\Big]
  `}</Latex
>
<p>
  There is a problem that we are facing now. We would like to use the data that
  we collected using the old policy <Latex>{String.raw`\pi_{\theta}`}</Latex> to
  optimize our new policy <Latex>{String.raw`\pi_{\theta'}`}</Latex>. But the
  dynamics <Latex>\rho</Latex> and the policy <Latex>\pi</Latex> need to be collected
  using the new parameters <Latex>\theta'</Latex>. We can deal with the policy
  using importance sampling.
</p>
<Latex
  >{String.raw`
    J(\theta') - J(\theta) = \sum_s \rho_{\pi_{\theta'}}(s) \sum_a \pi_{\theta}(a) \Big[\dfrac{\pi_{\theta'}}{\pi_{\theta}}A^{\theta}(s, a)\Big]
  `}</Latex
>
<p>
  On the other hand we have no way of dealing with <Latex>\rho</Latex>, because
  that would require sampling with the new policy and our goal is to use samples
  from the old policy. The idea is to assume<Latex
    >{String.raw`\rho_{\theta'} \approx \rho_{\theta}`}</Latex
  >. As it turns out this is only valid when the policy<Latex
    >{String.raw`\pi_{\theta}`}</Latex
  > and<Latex>{String.raw`\pi_{\theta'}`}</Latex> are close to each other. The authors
  of TRPO decided to use KL divergence as their measure of closeness, so that <Latex
    >{String.raw`D_{KL}(\pi_{\theta'} || \pi_{\theta} ) < \epsilon`}</Latex
  >. That we define a "trust region", where the difference between the two
  distributions is close and it is safe to take a gradient ascent step.
</p>

<Latex
  >{String.raw`
    \max_{\theta} {E}_{\theta} \Big[\dfrac{\pi_{\theta'}}{\pi_{\theta}}\hat{A}^{\theta}\Big] \text{, such that } D_{KL}(\pi_{\theta'} || \pi_{\theta} ) < \epsilon
  `}</Latex
>
<p>
  The optimization problem that we end up with looks as above. When <Latex
    >\theta</Latex
  > and <Latex>\theta'</Latex> are equal the KL divergence is 0 and the left hand
  side corresponds to an actor-critic optimization problem. When as in TRPO this
  is not the case the left side is called the "surrogate loss", because we approximate
  the loss function that requires sampling using the new policy through samples from
  the old policy. We do in order to use old samples several times before we discard
  them. Additionally the KL divergence creates a "trust region" that does not allow
  the new policy to stray away from the old policy. The whole process makes TRPO
  a more stable algorithm than the usual state of the art actor-critic methods.
</p>
<p>
  We are not going to further discuss how the optimization problem can be solved
  and how the algorithm exactly looks like. This is because on the one hand the
  derivations in TRPO are quite involved and on the other hand TRPO was used as
  a stepping stone for PPO. PPO is a much easier and popular algorithm, that
  builds on the foundations that we have discussed so far. Unlike TRPO it can be
  implemented using the standard autodiff libraries without additional overhead
  and has been used to create state of the art results.
</p>
<div class="separator" />
