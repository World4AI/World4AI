<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";
  import Latex from "$lib/Latex.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";

  let header = ["State", "Value"];
  let data = [
    [0, 1],
    [1, 2],
    [2, 1.5],
    [3, 3],
  ];
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | State and Value Representation</title
  >
  <meta
    name="description"
    content="When using approximative value functions, the state is represented by a feature vector and the value function takes that vector as an input and produces a state or action value."
  />
</svelte:head>

<h1>State and Value Representation</h1>
<Question
  >How can we represent states and value functions in environment with
  continuous state space?</Question
>
<div class="separator" />
<h2>State Representation</h2>
<p>
  So far each state was represented by a single number. This number was used as
  an address in the state-value or action-value lookup table.
</p>
<Table {header} {data} />
<p>
  For complex Markov decision processes that approach is not sustainable,
  because for most interesting problems the number of states is either larger
  than the number of atoms in the observable universe or states are continuous.
  In those cases the state is represented by a so-called feature vector. Each
  number in the vector gives some information about the state, while the whole
  feature vector is the representation of the state. In many cases the
  representation is going to be only partial and depict an observation (and not
  a state) in a partially observable Markov decision process.
</p>
<p>
  We write the feature vector as a column vector consisting of <Latex>n</Latex> features.
</p>
<Latex
  >{String.raw`
\mathbf{x} \doteq 
\begin{bmatrix}
  x_1 \\
  x_2 \\ 
  x_3 \\
  \cdots \\
  x_n 
\end{bmatrix}
  `}</Latex
>
<p>
  In the Cart Pole environment the feature vector consists of cart position,
  cart velocity, pole angle and angular velocity.
</p>
<Latex
  >{String.raw`
\mathbf{x} = 
\begin{bmatrix}
  \text{Cart Position} \\
  \text{Cart Velocity} \\ 
  \text{Pole Angle} \\
  \text{Angular Velocity} \\
\end{bmatrix}
  `}</Latex
>
<h2>Value Function Representation</h2>
<p>
  The feature vector <Latex>{String.raw`\mathbf{x}`}</Latex> is used as an input
  for the value function and the output is a single state-value or action-value number.
  The value function transforms the feature vector using an additional vector, called
  the weight vector <Latex>{String.raw`\mathbf{w} \in \mathbb{R}^n`}</Latex>.
  How exactly the weights are used in the calculation depends on the type of the
  function, but the general notation for function approximators is
  <Latex>{String.raw`\hat{v}(s, \mathbf{w})`}</Latex> for state-value functions and
  <Latex>{String.raw`\hat{q}(s, a, \mathbf{w})`}</Latex> for action-value functions.
  Where the "^" above the function (read as hat) shows that the function is an approximation
  and the weight vector <Latex>{String.raw`\mathbf{w} `}</Latex> shows that the calculation
  of state or action values requires that vector.
</p>
<p>
  There are many different types of function approximators that can be used to
  estimate the value of a state:
</p>
<ul>
  <li>Linear Function Approximators</li>
  <li>Neural Networks</li>
  <li>Decision Trees</li>
  <li>...</li>
</ul>

<p>
  At the moment of writing most modern reinforcement learning function
  approximators are neural networks. Linear function approximators are
  especially useful to introduce the topic of function approximators, as those
  are easiest to grasp and show some useful mathematical properties. We will
  start with linear approximators to clarify notation and the general
  functionality of approximative value functions, but mostly focus on neural
  networks in future chapters.
</p>

<p>
  Linear function approximators essentially calcualte the dot product between
  the feature vector and the weight vector to produce the value of the state.
</p>
<Latex
  >{String.raw`\hat{v}(s, \mathbf{w}) \doteq \mathbf{w} \cdot \mathbf{x} =  \mathbf{w}^T\mathbf{x} = w_1x_1 + w_2x_2 + \cdots + w_nx_n`}</Latex
>
<p>
  Neural networks on the other hand take the vector <Latex
    >{String.raw`\mathbf{x}`}</Latex
  > as the input and apply several linear transformations and non-linear activation
  functions in succession until a single state or action value is produced.
</p>
<NeuralNetwork />

<p>
  The challenge of reinforcement learning is to find the correct weight vector <Latex
    >{String.raw`\mathbf{w}`}</Latex
  > which generates values that are as close as possible to the true values <Latex
    >v(s)</Latex
  > for the policy <Latex>\pi</Latex>.
</p>
<div class="separator" />

<style>
  li {
    font-size: 20px;
    margin-left: 20px;
  }
</style>
