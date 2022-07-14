<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Dynamic Programming</title>
  <meta
    name="description"
    content="In Dynamic programming we have access to the model of the finite Markov decision process and can use iterative planning techniques to find the optimal value function and optimal policy."
  />
</svelte:head>

<h1>Dynamic Programming</h1>
<Question
  >Are dynamic programming algorithms learning or planning algorithms?</Question
>
<div class="separator" />
<p>
  We can compare different policies by comparing their respective value
  functions <Latex>{String.raw`v_{\pi}(s)`}</Latex>. For example if the value
  function of one policy is always larger than the value function of the second
  policy we can state that the first policy is strictly better than the second
  policy:
  <Highlight
    ><Latex>{String.raw`v_{\pi} > v_{\pi'} \Rightarrow \pi > \pi'`}</Latex
    ></Highlight
  >. It seems therefore logical that if our goal is to find the optimal policy,
  our first priority should be to find a way to compute value functions for a
  given policy. To put it differently: we have to find the solution to the
  following Bellman equation.
</p>
<Latex
  >{String.raw`
    \begin{aligned}
   v_{\pi}(s)  & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(s') \mid S_t=s] \\
& = \sum_a \pi(a \mid s)  R(a, s) + \gamma \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)v_{\pi}(s')
\end{aligned}
`}</Latex
>
<div class="separator" />

<h2>Closed Form Solutions</h2>
<p>
  If we know the model of the environment, <Latex>R(a, s)</Latex> and <Latex
    >P(s' | s, a)</Latex
  >, we can create a system of equations with <Latex>N</Latex> unknowns and <Latex
    >N</Latex
  > equations, where <Latex>N</Latex> is the number of states in the environment.
</p>
<p>
  To simplify notation we average the reward function <Latex>R(a,s)</Latex> and the
  transition function <Latex>P(s' \mid s, a)</Latex> over the possible actions from
  the policy <Latex>\pi</Latex> and create the following definitions.
</p>
<Latex>{String.raw`R_{\pi}(s)\doteq \sum_a \pi(a \mid s) R(a, s) `}</Latex>
<Latex
  >{String.raw`P_{\pi}(s \mid s')\doteq \sum_a \pi(a \mid s) P(s' \mid a, s) `}</Latex
>
<p>
  Finally we can set up the system of equations, that we need to solve
  simultaneously.
</p>
<Latex
  >{String.raw`
\begin{bmatrix}
  v_{\pi}(1) \\
  \vdots \\
  v_{\pi}(N)
\end{bmatrix}
  = 
\begin{bmatrix}
  R_{\pi}(1) \\
  \vdots \\
  R_{\pi}(N) \\
\end{bmatrix}
+ \gamma
\begin{bmatrix}
  P(1 \mid 1) & \dots & P(1 \mid N) \\
  \vdots & \vdots & \vdots \\
  P(N \mid 1) & \dots & P(N \mid N) \\
\end{bmatrix}
\begin{bmatrix}
  v_{\pi}(1) \\
  \vdots \\
  v_{\pi}(N)
\end{bmatrix}
`}</Latex
>
<Latex
  >{String.raw`
\begin{aligned}
  \mathbf{v_{\pi}} &= \mathbf{R_{\pi}} + \gamma \mathbf{P_{\pi}} \mathbf{v_{\pi}} \\
  \mathbf{v_{\pi}} -  \gamma \mathbf{P_{\pi}} \mathbf{v_{\pi}} & = \mathbf{R_{\pi}} \\
  (\mathbf{I} - \gamma \mathbf{P_{\pi}}) \mathbf{v_{\pi}} & = \mathbf{R_{\pi}} \\
  \mathbf{v_{\pi}} & = (\mathbf{I} - \gamma \mathbf{P_{\pi}})^{-1} \mathbf{R_{\pi}} \\
\end{aligned}
    `}</Latex
>
<p>
  The above solution is mathematically correct, but challenging in many relevant
  aspects. First, the inverse of the matrix introduces computational complexity
  of the magnitute <Latex>O(N^3)</Latex>. The cubic growth makes any practical
  algorithm with a sufficient large state space infeasible. Second, the above
  system of equations allows us to find the value function for a given policy <Latex
    >\pi</Latex
  >, but it not clear how we could find the optimal policy and value function
  for a given Markov decision process. Dynamic programming alleviates both of
  the above mentioned problems.
</p>
<div class="separator" />
<h2>Dynamic Programming Solutions</h2>
<p>
  Just as the Bellman equations, dynamic programming was developed by Richard
  Bellman in the 1950's. The idea of dynamic programming is to divide a problem
  into a collection of subproblems that can be solved recursively.
</p>
<p>
  A common example that is used to demonstrate the usefulness of dynamic
  programming is to show how the n'th member in the Fibonacci series can be
  found. The numbers in the Fibonacci series are always the sum of the previous
  two numbers in the series and the very first numbers are 0 and 1. We define
  the function that returns the n'th number in the series as <Latex>F(n)</Latex
  >, where for example <Latex>F(0) = 0</Latex> and <Latex>F(5) = 5</Latex>.
</p>
<Latex>0, 1, 1, 2, 3, 5, 8, 13, 21, 34 \dots</Latex>
<p>
  Looking at the sequence of numbers above we can confirm that the Fibonacci
  series is recursive: <Latex>F(n) = F(n-1) + F(n-2)</Latex>. For example:
</p>
<Latex>
  {String.raw`
\begin{aligned}
F(5) & = F(4) + F(3) \\
5 & = 3 + 2
\end{aligned}

  `}
</Latex>
<p>
  Using the same logic: <Latex>F(n-1) = F(n-2) + F(n-3)</Latex>. At this point
  in time it becomes clear that <Latex>F(n-2)</Latex> is needed to calculate <Latex
    >F(n)</Latex
  >
  and <Latex>F(n-1)</Latex>, which in computer science is called
  <em>overlapping subproblems</em>. Therefore we only need to calculate the
  value of <Latex>F(n-2)</Latex> once, store the value in memory and each time we
  need to have access to that value we access the stored value instead of recalculating.
</p>
<svg version="1.1" viewBox="0 0 1300 550" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="marker28944" overflow="visible" orient="auto">
      <path
        transform="scale(.8)"
        d="m5.77 0-8.65 5v-10z"
        fill="var(--text-color)"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
    <marker id="TriangleOutL" overflow="visible" orient="auto">
      <path
        transform="scale(.8)"
        d="m5.77 0-8.65 5v-10z"
        fill="var(--text-color)"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g
    id="calculated"
    stroke="#000"
    fill="var(--main-color-1)"
    stroke-dasharray="1.43111, 1.43111"
    stroke-linecap="round"
    stroke-width="1.4311"
  >
    <rect x="710" y="9.5" width="140" height="52.5" />
    <rect x="370" y="127" width="140" height="52.5" />
    <rect x="210" y="244.5" width="140" height="52.5" />
    <rect x="94" y="362" width="140" height="52.5" />
  </g>
  <g
    id="optimized"
    stroke="#000"
    fill="var(--main-color-2)"
    stroke-dasharray="1.43111, 1.43111"
    stroke-linecap="round"
    stroke-width="1.4311"
  >
    <rect x="1020" y="127" width="140" height="52.5" />
    <rect x="550" y="244.5" width="140" height="52.5" />
    <rect x="923.33" y="244.5" width="140" height="52.5" />
    <rect x="1150" y="244.5" width="140" height="52.5" />
    <rect x="834" y="362" width="140" height="52.5" />
    <rect x="1010" y="362" width="140" height="52.5" />
    <rect x="474" y="362" width="140" height="52.5" />
    <rect x="650" y="362" width="140" height="52.5" />
    <rect x="270" y="362" width="140" height="52.5" />
    <rect x="10" y="479.5" width="140" height="52.5" />
    <rect x="170" y="479.5" width="140" height="52.5" />
  </g>
  <g id="text" fill="black" font-family="sans-serif" font-size="40px">
    <text
      x="739.92188"
      y="48.289062"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="739.92188" y="48.289062">F(5)</tspan></text
    >
    <text
      x="399.92188"
      y="165.78905"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="399.92188" y="165.78905">F(4)</tspan></text
    >
    <text
      x="239.92188"
      y="283.28903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="239.92188" y="283.28903">F(3)</tspan></text
    >
    <text
      x="123.92184"
      y="400.78903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="123.92184" y="400.78903">F(2)</tspan></text
    >
    <text
      x="1049.9219"
      y="165.78905"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="1049.9219" y="165.78905">F(3)</tspan></text
    >
    <text
      x="579.92188"
      y="283.28903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="579.92188" y="283.28903">F(2)</tspan></text
    >
    <text
      x="953.25519"
      y="283.28903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="953.25519" y="283.28903">F(2)</tspan></text
    >
    <text
      x="1179.9219"
      y="283.28903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="1179.9219" y="283.28903">F(1)</tspan></text
    >
    <text
      x="863.92188"
      y="400.78903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="863.92188" y="400.78903">F(1)</tspan></text
    >
    <text
      x="1039.9219"
      y="400.78903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="1039.9219" y="400.78903">F(0)</tspan></text
    >
    <text
      x="503.92188"
      y="400.78903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="503.92188" y="400.78903">F(1)</tspan></text
    >
    <text
      x="679.92188"
      y="400.78903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="679.92188" y="400.78903">F(0)</tspan></text
    >
    <text
      x="299.92188"
      y="400.78903"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="299.92188" y="400.78903">F(1)</tspan></text
    >
    <text
      x="39.921875"
      y="518.28906"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="39.921875" y="518.28906">F(1)</tspan></text
    >
    <text
      x="199.92188"
      y="518.28906"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="199.92188" y="518.28906">F(0)</tspan></text
    >
  </g>
  <g id="arrows" fill="none" stroke="var(--text-color)" stroke-width="2px">
    <path d="m710 32-280 80" marker-end="url(#TriangleOutL)" />
    <path d="m850 32 240 80" marker-end="url(#marker28944)" />
    <path d="m430 192-160 40" marker-end="url(#marker28944)" />
    <path d="m450 192 180 40" marker-end="url(#TriangleOutL)" />
    <path d="m1090 192-100 40" marker-end="url(#TriangleOutL)" />
    <path d="m1130 192 100 40" marker-end="url(#marker28944)" />
    <path d="m270 312-100 40" marker-end="url(#marker28944)" />
    <path d="m290 312 60 40" marker-end="url(#marker28944)" />
    <path d="m150 432-80 40" marker-end="url(#marker28944)" />
    <path d="m170 432 80 40" marker-end="url(#marker28944)" />
    <path d="m610 312-80 40" marker-end="url(#marker28944)" />
    <path d="m630 312 100 40" marker-end="url(#marker28944)" />
    <path d="m970 312-60 40" marker-end="url(#marker28944)" />
    <path d="m1010 312 60 40" marker-end="url(#marker28944)" />
  </g>
</svg>
<p>
  The image above shows how the dynamic programming algorithm can be used to
  efficiently calculate the numbers in the Fibonacci sequence. The reddish
  numbers indicate function calls that need to actually calculate the sum of two
  numbers in the sequence. The blue numbers indicate function calls that look up
  the values in memory and therefore do not require additional computations. As
  the number in the sequence, <Latex>n</Latex>, becomes larger the computation
  savings become larger and larger.
</p>
<p>
  We can argue that dynamic programming can also be applied to Bellman
  equations, because the Bellman equations are structured in such a way, that
  the value of a certain state <Latex>s</Latex> depends on values of other states
  <Latex>s'</Latex>.
</p>
<Latex
  >{String.raw`
   v_{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(s') \mid S_t=s] \\
`}</Latex
>
<div class="container">
  <svg version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="TriangleOutL" overflow="visible" orient="auto">
        <path
          transform="scale(.8)"
          d="m5.77 0-8.65 5v-10l8.65 5z"
          fill="context-stroke"
          fill-rule="evenodd"
          stroke="context-stroke"
          stroke-width="1pt"
        />
      </marker>
    </defs>
    <g
      stroke="black"
      fill="var(--text-color)"
      stroke-linecap="round"
      stroke-width="1.2267"
    >
      <rect x="200" y="20" width="100" height="60" />
      <rect x="200" y="420" width="100" height="60" />
      <rect x="380" y="220" width="100" height="60" />
      <rect x="20" y="220" width="100" height="60" />
    </g>
    <g fill="var(--background-color)" font-family="sans-serif" font-size="40px">
      <text
        x="210.95703"
        y="62.539062"
        style="line-height:1.25"
        xml:space="preserve"
        ><tspan x="210.95703" y="62.539062">v(0)</tspan></text
      >
      <text
        x="210.95703"
        y="462.53906"
        style="line-height:1.25"
        xml:space="preserve"
        ><tspan x="210.95703" y="462.53906">v(2)</tspan></text
      >
      <text
        x="390.95703"
        y="262.53906"
        style="line-height:1.25"
        xml:space="preserve"
        ><tspan x="390.95703" y="262.53906">v(1)</tspan></text
      >
      <text
        x="30.957031"
        y="262.53906"
        style="line-height:1.25"
        xml:space="preserve"
        ><tspan x="30.957031" y="262.53906">v(3)</tspan></text
      >
    </g>
    <g fill="none" stroke="var(--text-color)" stroke-width="1px">
      <path d="m250 100 180 100" marker-end="url(#TriangleOutL)" />
      <path d="m430 300-180 100" marker-end="url(#TriangleOutL)" />
      <path d="m270 400 180-100" marker-end="url(#TriangleOutL)" />
      <path d="m70 200 170-110" marker-end="url(#TriangleOutL)" />
      <path d="m220 90-170 110" marker-end="url(#TriangleOutL)" />
      <path d="m450 200-180-100" marker-end="url(#TriangleOutL)" />
      <path d="m310 30h30v40h-30" marker-end="url(#TriangleOutL)" />
      <path d="m130 230h30v40h-30" marker-end="url(#TriangleOutL)" />
    </g>
  </svg>
</div>
<p>
  The subproblems in Bellman equations can be infinitely more comples than those
  in the Fibonacci sequence. The reason for that is the complexity of
  interconnections between the subproblems. In the drawing for example the value
  of state 0 depends on the value of state 1 and the value of state 1 depends on
  the value of state 0. At the same time value of state 0 depends on itself.
</p>
<p>
  The dynamic programming algorithms that we are going to cover in this section
  are not those that are commonly used to solve reinforcement learning tasks. In
  fact there is no learning involved at all. Dynamic programming (DP) requires
  the full knowledge of the model of the environment and calculates the optimal
  value function and optimal policy through the knowledge of that model. The
  interaction between the agent and the environment is not necessary.
  Nevertheless dynamic programming is the usual introduction to the solution of
  MDPs, because the knowledge gained by studying DP algorithms is transferable
  to reinforcement learning.
</p>
<div class="separator" />

<style>
  .container {
    margin: 0 auto;
    width: 400px;
  }
</style>
