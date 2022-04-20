<script>
  import Question from "$lib/Question.svelte";
  import Lineplot from "$lib/Lineplot.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

  let data = [];
  let points = [];
  let startingPoint = [];
  let lines = [];

  let pointX = 55;
  $: pointY = pointX ** 2;
  $: m = 2 * pointX;
  let alpha = 0.01;

  startingPoint.push({ x: 55, y: 55 ** 2 });

  // draw parabola x^2
  for (let i = -60; i <= 60; i++) {
    let x = i;
    let y = x ** 2;
    data.push({ x, y });
  }

  function recalculatePoints() {
    points = [];
    points.push({ x: pointX, y: pointY });
  }

  function drawSlope() {
    // draw slope
    //derivative is 2x
    //the equation is y = mx + t, and we have m (2x) and y+x come from the current point
    //therefore we can find t and draw the line
    let m = 2 * pointX;
    let t = pointY - m * pointX;
    lines = [];
    lines.push({
      x1: -100,
      y1: m * -100 + t,
      x2: 100,
      y2: m * 100 + t,
    });
  }

  function gradientDescentStep() {
    pointX = pointX - alpha * 2 * pointX;
  }

  $: pointX && recalculatePoints();
  $: pointX && drawSlope();
</script>

<h1>Gradient Descent</h1>
<Question
  >How can we learn the optimal weights and biases in linear regression?</Question
>
<div class="separator" />

<h2>Gradient Descent From A Known Function</h2>
<p>
  Before we discuss the algorithm that can find the optimal weights and the bias
  for linear regression, let us take a step back and consider how we can find
  the value for <Latex>x</Latex> that minimizes
  <Latex>f(x) = x^2</Latex>. In other words we know exactly how the function <Latex
    >f(x)</Latex
  > looks like.
</p>

<p>
  The equation depicts a parabola and from visual inspection we can determine,
  that the <Latex>x</Latex> value of 0 produces the minimum <Latex>f(x)</Latex>.
</p>
<Lineplot
  {data}
  minX={-80}
  maxX={80}
  minY={0}
  maxY={3000}
  numTicks={5}
  xLabel={"x"}
  yLabel={"f (x)"}
/>
<p>
  In machine learning we rarely have the luxury of being able to visually find
  the optimal solution. Our functions are usually dependend on thousand or
  million features and that is something that we can not visualize. We therefore
  start with a random <Latex>x</Latex> value. In the example below we pick 55.
</p>
<Lineplot
  {data}
  points={startingPoint}
  minX={-80}
  maxX={80}
  minY={0}
  maxY={3000}
  numTicks={5}
  xLabel={"x"}
  yLabel={"f(x)"}
/>
<p>
  Next we calculate the derivative of <Latex>f(x)</Latex> with respect to <Latex
    >x</Latex
  >. Using the basic rules of calculus we derive <Latex
    >{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x) = 2x`}</Latex
  >. For our starting value that means that the slope for our starting value is
  110. We can draw the line at the starting point to visualize the derivative.
</p>
<Lineplot
  {data}
  {points}
  {lines}
  minX={-80}
  maxX={80}
  minY={0}
  maxY={3000}
  numTicks={5}
  xLabel={"x"}
  yLabel={"f(x)"}
/>
<p>
  The derivative shows us the direction in which we can shift <Latex>x</Latex>,
  to be exact it gives us the direction of the steepes descent and ascent. When
  we want to minimize <Latex>f(x)</Latex> we use gradient descent and subtract <Latex
    >{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex
  > from <Latex>x</Latex>.
</p>
<Latex
  >{String.raw`\large x_{t+1} \coloneqq x_t - \alpha \dfrac{\mathrm{d}}{\mathrm{d}x}f(x_t)`}</Latex
>
<p>
  Essentially at each timestep we update the variable <Latex>x</Latex> until we get
  close to <Latex>0</Latex>. While the derivative gives us the direction in
  which should take a step, the derivative does not give us the size of the
  step. For that purpose we use a variable <Latex>\alpha</Latex> also called the
  <Highlight>learning rate</Highlight>. The learning rate scales the derivative
  by multiplying the direction with a value that usually lies between 0.1 and
  0.001. That prevents the algorithm from diverging.
</p>
<p>
  Below is an interactive example that is designed to get the intuition for
  gradient descent.
</p>

<Lineplot
  {data}
  {points}
  {lines}
  minX={-80}
  maxX={80}
  minY={0}
  maxY={3000}
  numTicks={5}
  xLabel={"x"}
  yLabel={"f(x)"}
/>
<form>
  <div class="form-group">
    <label for="alpha"><Latex>\alpha</Latex></label>
    <input
      on:keydown|preventDefault
      id="alpha"
      bind:value={alpha}
      min="0"
      step="0.01"
      type="number"
    />
  </div>
  <div class="form-group">
    <label for="alpha"
      ><Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex></label
    >
    <div class="form-box">{m}</div>
  </div>
  <div class="form-group">
    <label for="x"><Latex>x</Latex></label>
    <input
      on:keydown|preventDefault
      id="x"
      bind:value={pointX}
      min="-80"
      max="80"
      step="1"
      type="number"
    />
  </div>
  <button on:click|preventDefault={gradientDescentStep}
    >Take Gradient Descent Step</button
  >
</form>

<p>
  You can learn several things if you play with the example. If you try positive
  and negative <Latex>x</Latex> values you will observe that the sign of the derivative
  changes based on the sign of the location of <Latex>x</Latex>. That behaviour
  makes sure that we distract negative values from <Latex>x</Latex> when <Latex
    >x</Latex
  > is negative and we distract positive values from <Latex>x</Latex> when <Latex
    >x</Latex
  > is positive. You could also try gradient descent with an <Latex
    >\alpha</Latex
  > of 1.01 and observe that the algorithm starts to diverge. Picking the correct
  learning rate comes is an extremely usefull skill and is generally on of the first
  things to tweak when you want your algorithm to perform better. In fact <Latex
    >\alpha</Latex
  > is one of the so called <Highlight>hyperparameters</Highlight>. A
  hyperparamter is a parameter that is set by the programmer and that influences
  the learning of the parameters that you are truly interested in (like <Latex
    >w</Latex
  > and <Latex>b</Latex>). A third thing that you will should notice is the
  decrease of the derivative when we start getting closer and closer to the
  optimal value. You can also observe that the slope of the tangent gets flatter
  and flatter. This natural behaviour makes sure that we take smaller and
  smaller steps as we start approaching the optimum. This also means that
  gradient descent does not find an optimal value for <Latex>x</Latex>, but if
  we are lucky one that is close enough. At this point the question might arise:
  "how do we know when to stop then?". A common approach in machine learning is
  to take steps for a certain amount of iterations. We will discuss the topic
  more throughout this book.
</p>

<p>
  Before we move on to the part where we discuss how we can apply this algorithm
  to linear regression, let us discuss how we can deal with functions that have
  more than one variable, for example <Latex
    >{String.raw`f(x_1, x_2) = x_1^2 + x_2^2`}</Latex
  >. The approach is actually very similar. Insdead of calculating the
  derivative with respect to <Latex>x</Latex>
  <Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex> we need to calculate
  the partial derivatives with respect to all variables, in our case
  <Latex>{String.raw`\dfrac{\mathrm{\delta}}{\mathrm{\delta}x_1}f`}</Latex>
  and <Latex>{String.raw`\dfrac{\mathrm{\delta}}{\mathrm{\delta}x_2}f`}</Latex>.
  For convenience we put the partial derivatives and the variables into their
  corresponding vectors.
</p>
<Latex>
  {String.raw`
\large
\mathbf{x} = 
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
  `}
</Latex>
<Latex>
  {String.raw`
\large
\mathbf{\nabla} = 
\begin{bmatrix}
\dfrac{\mathrm{\delta}}{\mathrm{\delta}x_1}f \\
\dfrac{\mathrm{\delta}}{\mathrm{\delta}x_2}f
\end{bmatrix}
  `}
</Latex>
<p>
  The gradient descent algorithm looks almost the same, but for the substitution
  for vectors.
</p>
<Latex
  >{String.raw`\large \mathbf{x}_{t+1} \coloneqq \mathbf{x}_t - \alpha \mathbf{\nabla} `}</Latex
>
<p>
  The vector that is represented by nabla <Latex>\nabla</Latex> is called the <Highlight
    >gradient</Highlight
  >, giving its name to the gradient descent (or ascent) algorithm.
</p>
<div class="separator" />

<h2>Gradient Descent From Data</h2>
<p>
  When we deal with linear regression (or machine learning in general), we are
  confronted with a slightly different problem. We do not try to find a minimum
  of a given function directly, rather we try to find parameters (weights and
  bias) for a linear function that best approximates the function that generated
  the data. In machine learning that process is also called data or curve
  fitting, because we search for a curve/function that best fits the data that
  we use in our training process. For that purpose we use gradient descent to
  minimize a loss function (the mean squared error in our case). We do not know
  how the real data generating function looks like, but we hope that by
  minimizing the loss function we can generate a relatively good approximation.
</p>
<h3>Batch Gradient Descent</h3>
<h3>Stochastic Gradient Descent</h3>
<h3>Mini-Batch Gradient Descent</h3>
<div class="separator" />

<style>
  input,
  .form-box,
  button {
    background-color: var(--background-color);
    outline: none;
    border: 1px solid var(--text-color);
    color: var(--text-color);
    padding: 10px;
    font-size: 15px;
    width: 100%;
  }
  button {
    cursor: pointer;
  }
  .form-group {
    margin-bottom: 10px;
    align-items: center;
    justify-content: center;
    width: 100%;
    display: flex;
  }
  label {
    display: block;
    margin-bottom: 8px;
    width: 100px;
  }

  button:hover {
    color: var(--main-color-1);
  }
</style>
