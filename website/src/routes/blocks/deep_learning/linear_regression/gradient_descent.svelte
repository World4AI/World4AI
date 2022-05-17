<script>
  import Container from "$lib/Container.svelte";
  import Lineplot from "$lib/Lineplot.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Mse from "./_loss/Mse.svelte";

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

  // gradient descent mse animation
  const dataMse = [
    [
      { x: 5, y: 20 },
      { x: 10, y: 40 },
      { x: 35, y: 15 },
      { x: 45, y: 59 },
    ],
  ];

  let w = 1;
  let b = 1;
  let numSamples = dataMse[0].length;
  let mse;
  let mseAlpha = 0.001;
  let numEpochs = 10;

  function calculateMse() {
    mse = 0;
    dataMse[0].forEach((sample) => {
      mse += (w * sample.x + b - sample.y) ** 2;
    });
    mse /= numSamples;
  }
  calculateMse();

  function mseGradientDescentStep() {
    for (let i = 0; i < numEpochs; i++) {
      let db = 0;
      let dw = 0;

      dataMse[0].forEach((sample) => {
        db += w * sample.x + b - sample.y;
        dw += sample.x * (w * sample.x + b - sample.y);
      });
      db /= numSamples;
      dw /= numSamples;
      w = w - mseAlpha * dw;
      b = b - mseAlpha * db;
      calculateMse();
    }
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Linear Regression Gradient Descent</title>
  <meta
    name="description"
    content="Gradient descent is the algorithm that is most commonly used to find the optimal weights and biases in linear regression."
  />
</svelte:head>

<Container>
  <h1>Gradient Descent</h1>
  <div class="separator" />

  <h2>Gradient Descent From A Known Function</h2>
  <p>
    Before we discuss the algorithm that can find the optimal weights and the
    bias for linear regression, let us take a step back and consider how we can
    find the value for <Latex>x</Latex> that minimizes
    <Latex>f(x) = x^2</Latex>. In other words we know exactly how the function <Latex
      >f(x)</Latex
    > looks like.
  </p>

  <p>
    The equation depicts a parabola and from visual inspection we can determine,
    that the <Latex>x</Latex> value of 0 produces the minimum <Latex>f(x)</Latex
    >.
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
    million features and that is something that we can not visualize. We
    therefore start with a random <Latex>x</Latex> value. In the example below we
    pick 55.
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
    >. For our starting value that means that the slope for our starting value
    is 110. We can draw the line at the starting point to visualize the
    derivative.
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
    The derivative shows us the direction in which we can shift <Latex>x</Latex
    >, to be exact it gives us the direction of the steepes descent and ascent.
    When we want to minimize <Latex>f(x)</Latex> we use gradient descent and subtract
    <Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex> from <Latex
      >x</Latex
    >.
  </p>
  <Latex
    >{String.raw`\large x_{t+1} \coloneqq x_t - \alpha \dfrac{\mathrm{d}}{\mathrm{d}x}f(x_t)`}</Latex
  >
  <p>
    Essentially at each timestep we update the variable <Latex>x</Latex> until we
    get close to <Latex>0</Latex>. While the derivative gives us the direction
    in which should take a step, the derivative does not give us the size of the
    step. For that purpose we use a variable <Latex>\alpha</Latex> also called the
    <Highlight>learning rate</Highlight>. The learning rate scales the
    derivative by multiplying the direction with a value that usually lies
    between 0.1 and 0.001. That prevents the algorithm from diverging.
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
      <input id="alpha" bind:value={alpha} min="0" step="0.01" type="number" />
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
        id="x"
        bind:value={pointX}
        min="-80"
        max="80"
        step="1"
        type="number"
      />
    </div>
    <button type="button" on:click|preventDefault={gradientDescentStep}
      >Take Gradient Descent Step</button
    >
  </form>

  <p>
    You can learn several things if you play with the example. If you try
    positive and negative <Latex>x</Latex> values you will observe that the sign
    of the derivative changes based on the sign of the location of <Latex
      >x</Latex
    >. That behaviour makes sure that we distract negative values from <Latex
      >x</Latex
    > when <Latex>x</Latex> is negative and we distract positive values from <Latex
      >x</Latex
    > when <Latex>x</Latex> is positive. You could also try gradient descent with
    an <Latex>\alpha</Latex> of 1.01 and observe that the algorithm starts to diverge.
    Picking the correct learning rate comes is an extremely usefull skill and is
    generally on of the first things to tweak when you want your algorithm to perform
    better. In fact <Latex>\alpha</Latex> is one of the so called <Highlight
      >hyperparameters</Highlight
    >. A hyperparamter is a parameter that is set by the programmer and that
    influences the learning of the parameters that you are truly interested in
    (like <Latex>w</Latex> and <Latex>b</Latex>). A third thing that you will
    should notice is the decrease of the derivative when we start getting closer
    and closer to the optimal value. You can also observe that the slope of the
    tangent gets flatter and flatter. This natural behaviour makes sure that we
    take smaller and smaller steps as we start approaching the optimum. This
    also means that gradient descent does not find an optimal value for <Latex
      >x</Latex
    >, but if we are lucky one that is close enough. At this point the question
    might arise: "how do we know when to stop then?". A common approach in
    machine learning is to take steps for a certain amount of iterations. We
    will discuss the topic more throughout this book.
  </p>

  <p>
    Before we move on to the part where we discuss how we can apply this
    algorithm to linear regression, let us discuss how we can deal with
    functions that have more than one variable, for example <Latex
      >{String.raw`f(x_1, x_2) = x_1^2 + x_2^2`}</Latex
    >. The approach is actually very similar. Insdead of calculating the
    derivative with respect to <Latex>x</Latex>
    <Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex> we need to calculate
    the partial derivatives with respect to all variables, in our case
    <Latex>{String.raw`\dfrac{\mathrm{\delta}}{\mathrm{\delta}x_1}f`}</Latex>
    and <Latex>{String.raw`\dfrac{\mathrm{\delta}}{\mathrm{\delta}x_2}f`}</Latex
    >. For convenience we put the partial derivatives and the variables into
    their corresponding vectors.
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
    The gradient descent algorithm looks almost the same, but for the
    substitution for vectors.
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
    confronted with a slightly different problem. We do not try to find a
    minimum of a given function directly, rather we try to find parameters
    (weights and bias) for a linear function that best approximates the function
    that generated the data. In machine learning that process is also called
    data or curve fitting, because we search for a curve/function that best fits
    the data that we use in our training process. For that purpose we use
    gradient descent to minimize a loss function (the mean squared error in our
    case). We do not know how the real data generating function looks like, but
    we hope that by minimizing the loss function we can generate a relatively
    good approximation.
  </p>
  <h3>Single Training Sample</h3>
  <p>
    Let us remind ourselves that our goal is to minimize the mean squared error
    <Latex
      >{String.raw`MSE=\frac{1}{m}\sum_i^m (\hat{y}^{(i)} - y^{(i)})^2`}</Latex
    >. In the definition of <Latex>MSE</Latex> we use the upperscript notation <Latex
      >{String.raw`y^{(i)}`}</Latex
    > to indicate that there are <Latex>m</Latex> samples in the dataset and <Latex
      >i</Latex
    > is the index of a particular sample. Yet before we move on to discussing how
    we can use gradient descent with multiple data points, let us ease into the calculations
    by assuming that we have one single sample <Latex>m=1</Latex> so that the mean
    squared error is reduced to a much simpler form
    <Latex>{String.raw` MSE=(\hat{y} - y)^2`}</Latex>. Generally speaking we
    want to find the weight vector <Latex>{String.raw`\mathbf{w}`}</Latex> and the
    bias scalar <Latex>b</Latex> that minimize the mean squared error betwen the
    single prediction <Latex>{String.raw`\hat{y}`}</Latex> and the true label <Latex
      >y</Latex
    >.
  </p>
  <Highlight>
    <Latex
      >{String.raw`\large \arg\min_{\mathbf{w}, b} MSE=(\hat{y} - y)^2`}</Latex
    >
    <br />
    <Latex
      >{String.raw`\large \arg\min_{\mathbf{w}, b} MSE=(\mathbf{w^Tx} + b - y)^2`}</Latex
    >
  </Highlight>
  <p>
    Just as in the examples above we will use gradient descent to find the
    minimim of the mean squared error. Yet there is a significant difference in
    the notation. What we need to optimize are the weights <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and the bias<Latex>{String.raw`b`}</Latex> and not the inputs <Latex
      >{String.raw`\mathbf{x}`}</Latex
    >. The vector <Latex>{String.raw`\mathbf{x}`}</Latex> contains <Latex
      >n</Latex
    >
    features of a single data point and we can not change the features, therefore
    we consider this vector as fixed.
  </p>
  <p>
    The computation of the gradient is slighly more complicated than the one we
    covered above, because we have to apply the chain rule. To simplify notation
    let us define the expression in the round brackets as <Latex>a</Latex>.
  </p>
  <Latex>{String.raw`\large a = \mathbf{w^Tx} + b - y`}</Latex>
  <p>That way we can rewrite <Latex>MSE</Latex> in the following way.</p>
  <Latex>{String.raw`\large MSE = a^2`}</Latex>
  <p>
    In order to be able to apply gradient descent we need to calculate partial
    derivatives with respect to each weight <Latex>w_j</Latex> and the bias <Latex
      >b</Latex
    >.
  </p>
  <p>
    We will not calculate the derivatives directly, but apply the chain rule by
    utilizing <Latex>a</Latex> defined above.
  </p>
  <Latex
    >{String.raw`\large \dfrac{\delta MSE}{\delta w_j} = \dfrac{\delta MSE}{\delta a} \dfrac{\delta a}{\delta w_j}`}</Latex
  >
  <Latex
    >{String.raw`\large \dfrac{\delta MSE}{\delta b} = \dfrac{\delta MSE}{\delta a} \dfrac{\delta a}{\delta b}`}</Latex
  >
  <p>
    Using basic rules of calculus we derive the following partial derivatives.
  </p>
  <Latex>{String.raw`\large \dfrac{\delta MSE}{\delta a} = 2a`}</Latex>
  <Latex>{String.raw`\large \dfrac{\delta a}{\delta w_j} = x_j`}</Latex>
  <Latex>{String.raw`\large \dfrac{\delta a}{\delta b} = 1`}</Latex>
  <p>
    By applying the chain rule we end up with the desired partial derivatives.
  </p>
  <Latex
    >{String.raw`
  \large 
  \begin{aligned}
  \dfrac{\delta MSE}{\delta w_j} & = \dfrac{\delta MSE}{\delta a} \dfrac{\delta a}{\delta w_j} \\
& = 2ax_j \\
& = 2x_j (\mathbf{w^Tx} + b - y)
  \end{aligned}
  `}</Latex
  >
  <Latex
    >{String.raw`
  \large 
  \begin{aligned}
  \dfrac{\delta MSE}{\delta b} & = \dfrac{\delta MSE}{\delta a} \dfrac{\delta a}{\delta b} \\
& = 2a * 1 \\
& = 2 (\mathbf{w^Tx} + b - y)
  \end{aligned}
  `}</Latex
  >
  <p>
    Often you will see that the mean squared error is divided by 2 in the
    definition.
  </p>
  <Latex
    >{String.raw`\large MSE=\dfrac{1}{2m}\sum_i^m (\hat{y}^{(i)} - y^{(i)})^2`}</Latex
  >
  <p>
    This is done for convenience. If you look at the partial derivatives above
    you will notice, that all of them contain the number 2. The division by 2
    therefore cancels the 2 in the derivative and makes the results more clean.
  </p>
  <Latex
    >{String.raw`
  \large 
  \begin{aligned}
  \dfrac{\delta MSE}{\delta w_j} & = \dfrac{\delta MSE}{\delta a} \dfrac{\delta a}{\delta w_j} \\
& = \dfrac{1}{2}2a * w_j \\
& = x_j (\mathbf{w^Tx} + b - y)
  \end{aligned}
  `}</Latex
  >
  <Latex
    >{String.raw`
  \large 
  \begin{aligned}
  \dfrac{\delta MSE}{\delta b} & = \dfrac{\delta MSE}{\delta a} \dfrac{\delta a}{\delta b} \\
& = \dfrac{1}{2}2a * 1 \\
& =  (\mathbf{w^Tx} + b - y)
  \end{aligned}
  `}</Latex
  >
  <p>
    This adjustent is perfectly legal. Remember that the derivatives merely
    determine the direction and not the size of our step. By scaling the
    derivative we do not change the direction. The size of the derivative on the
    other hand will be determined by the learning rate.
  </p>
  <p>The gradient descent algorithm works as expected.</p>
  <Latex
    >{String.raw`\large \mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w \\`}</Latex
  >
  <Latex
    >{String.raw`\large b_{t+1} \coloneqq b_t - \alpha \dfrac{\delta}{\delta b} `}</Latex
  >
  <p>
    While didactically it makes sense to learn how we can calculate the gradient
    for a single sample first, practically we always deal with much larger
    datasets, often consiting of many thousands or even millions of samples. In
    practice we use batch, stochastic or mini-batch gradient descent.
  </p>
  <h3>Batch Gradient Descent</h3>
  <p>
    As it turns out making a jump from one to several samples is not that
    complicated. You should remember from calculus that the derivative of a sum
    is the sum of derivatives. In other words in order to calculate the gradient
    for the MSE below, we need to calculate the individual gradients for each
    sample, add those gradients and divide by the number of samples <Latex
      >m</Latex
    >.
  </p>
  <Highlight>
    <Latex
      >{String.raw`
\begin{aligned}
\large 
& MSE=\dfrac{1}{2m}\sum_i^m (\hat{y}^{(i)} - y^{(i)})^2 \\
& \mathbf{\nabla}_{w} = \dfrac{1}{m} \sum_i^m\mathbf{\nabla}^{(i)}_w \\
& \dfrac{\delta}{\delta b} = \dfrac{1}{m}\sum^m_i\dfrac{\delta}{\delta b}^{(i)}
\end{aligned}
    `}</Latex
    >
  </Highlight>
  <p>
    This approach of using the whole dataset for gradient descent is called <Highlight
      >batch</Highlight
    > gradient descent. The advantage of this approach are the great parallelization
    opportunities, because each of the gradients can be computed independently. Yet
    in many cases batch gradient descent is not used in practice. We often have to
    deal with datasets consisting of thousands of features and millions of samples.
    It is not possible to load all that data on the GPU's. The alternatives describe
    below are more practical.
  </p>

  <p>
    Below is the interactive example from the last section. This time we use
    batch gradient descent to find the optimal weight and bias. As we have only
    4 datapoints, batch gradient descent is a fine choice. You can adjust the
    learning rate and the number of epochs. This time we can reduce the mean
    squared error below 250.
  </p>

  <Mse data={dataMse} {w} {b} />
  <form>
    <div class="form-group">
      <label for="alpha-w"><Latex>\alpha</Latex></label>
      <input
        id="alpha-w"
        bind:value={mseAlpha}
        min="0"
        step="0.01"
        type="number"
      />
    </div>
    <div class="form-group">
      <label for="epochs">Epochs</label>
      <input
        id="epochs"
        bind:value={numEpochs}
        min="0"
        step="100"
        type="number"
      />
    </div>
    <div class="form-group">
      <label for="w"><Latex>{String.raw`w`}</Latex></label>
      <div id="w" class="form-box">{w}</div>
    </div>
    <div class="form-group">
      <label for="b"><Latex>{String.raw`b`}</Latex></label>
      <div id="b" class="form-box">{b}</div>
    </div>
    <div class="form-group">
      <label for="mse"><Latex>{String.raw`MSE`}</Latex></label>
      <div id="mse" class="form-box">{mse}</div>
    </div>
    <button type="button" on:click|preventDefault={mseGradientDescentStep}
      >Take Gradient Descent Step</button
    >
  </form>

  <h3>Stochastic Gradient Descent</h3>
  <p>
    In stochastic gradient descent we introduce some stochasticity by shuffling
    the dataset randomly and using one sample at a time until we have used all
    samples for training. This period of time, in which we exhaust all samples
    in training is called an <Highlight>epoch</Highlight>. After each epoch we
    reshuffle and start over. The advantage of stochastic gradient descent is
    that we do not have to wait for the calculation of gradients for all
    samples. We therefore take a gradient descent step after calculating the
    gradient for a single sample. But we lose the ability of parallelization.
  </p>

  <h3>Mini-Batch Gradient Descent</h3>
  <p>
    Mini-batch gradient descent combines the advantages of the stochastic and
    batch gradient descent. At the start of each epoch the dataset is shuffled
    randomly, but insdead of using one sample at a time in mini-batch gradient
    descent several samples are taken. Similar to the learning rate the size of
    the mini-batch is a hyperparameter and needs to be determined by the
    developer. Usually the size is calculated as a power of 2, for example 32,
    64, 128 and so on. You just need to remember that if needs to fit into the
    memory of your graphics card.
  </p>
  <p>
    Mini-batch gradient descent can be parallelized, because we use several
    samples at a time. Additionally it has the advantage that theoretically our
    training dataset can be as large as we want it to be.
  </p>
  <div class="separator" />
</Container>

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

  input[type="number"] {
    -moz-appearance: textfield;
  }

  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    opacity: 0;
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
