<script>
  import Table from "$lib/Table.svelte";
  import Latex from "$lib/Latex.svelte";

  let header = ["State", "Value"];
  let data = [
    [0, 1],
    [1, 2],
    [2, 1.5],
    [3, 3],
  ];
</script>

<h2>State Representation</h2>
<p>
  So far for finite MDPs each state was represented by a single number. This
  number was used as an address in the state-value or action-value lookup table.
</p>
<Table {header} {data} />
<p>
  In the example above to get the the state-value for state 3 for a certain
  policy \pi the agent looked at the value in the lookup table to receive the
  value of 3.
</p>
<p>
  For complex MDPs that approach is not sustainable, as for most interesting
  problems the number of states is larger than the number of atoms in the
  observable universe. Therefore a state is represented by a so-called feature
  vector. Each number in the vector gives some information about the state. The
  whole vector is the representation of the state. In many cases the
  representation is only partial, therefore in approximative methods we are
  going to use the word observation instead of state to show the possible
  limitations of state representations.
</p>
<Latex>{String.raw`\mathbf{x} \doteq (x_1(s), x_2(s), ... , x_d(s))^T`}</Latex>
<p>
  In the Cart Pole environment the feature vector consists of cart position,
  cart velocity, pole angle and angular velocity.
</p>
<Latex
  >{String.raw`\mathbf{x} \doteq (CartPosition, CartVelocity, PoleAngle, AngularVelocity)^T`}</Latex
>
<h2>Value Representation</h2>
<p>
  The feature vector <Latex>{String.raw`\mathbf{x}`}</Latex> is used as an input
  into the approximative value function and the output is a single state-value or
  action-value number.
</p>
<p>
  In order for the value function to transform the feature vector into the
  single number representation an additional vector, called weight vector,
  <Latex>{String.raw`\mathbf{w} \in \mathbb{R}^n`}</Latex> is needed. How exactly
  the weights are used in the calculation depends on the type of the function.
</p>
<p>There are many different types of function approximators:</p>
<ul>
  <li>Linear Function Approximators</li>
  <li>Neural Networks</li>
  <li>Decision Trees</li>
  <li>...</li>
</ul>

<p>
  Depending on the function approximators the weight vector might play a
  different role in the calculation of the value function, but the general way
  to write down function approximators is as follows.
</p>
<Latex>{String.raw`\hat{v}(s, \mathbf{w})`}</Latex>
<Latex>{String.raw`\hat{q}(s, a, \mathbf{w})`}</Latex>
<p>
  Where the "^" above the function (read as hat) shows that the function is an
  approximation and the weight vector <Latex>{String.raw`\mathbf{w} `}</Latex> shows
  that the calculation of state or action values requires that vector.
</p>
<p>
  At the moment of writing most modern reinforcement learning function
  approximators are neural networks. Linear function approximators are
  especially useful to introduce the topic of function approximators, as those
  are easiest to grasp and show some useful mathematical properties.
</p>

<p>
  In linear function approximators each of the features is “weighted” by the
  corresponding weight. The individual weighted features are summed up to
  produce the value.
</p>
<Latex
  >{String.raw`\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^T\mathbf{x}(s) \doteq \sum_{i=1}^d w_i x_i(s)`}</Latex
>

<p>
  Let us again look at the Cart Pole environment to clarify the linear function
  approximation. Below is one of the possible initial values for the feature
  vector.
</p>
<Latex
  >{String.raw`\mathbf{x} = [0.04371849, -0.04789172, -0.03998533, -0.01820894]`}</Latex
>
<p>
  In order to calculate the approximate state value for a particular <Latex
    >\pi</Latex
  > for the above state the following equation has to be calculated.
</p>
<Latex
  >{String.raw`\hat{v}(s, \mathbf{w}) = w_1 * 0.04371849 + w_2 * (-0.04789172) + w_3 * (-0.03998533) + w_4 * (-0.01820894)`}</Latex
>

<p>
  The same four weights are used for the calculation of the state value for all
  possible feature vectors.
</p>

<p>
  Neural networks are non-linear function approximators, where each neuron in
  itself is a non-linear function. The calculation for each neuron is similar to
  that of the linear function, but the result of the weighted sum is used as an
  input to a non-linear function f().
</p>
<Latex
  >{String.raw`\hat{v}(s, \mathbf{w}) \doteq f(\mathbf{w}^T\mathbf{x}(s)) \doteq f(\sum_{i=1}^d w_i x_i(s))`}</Latex
>
