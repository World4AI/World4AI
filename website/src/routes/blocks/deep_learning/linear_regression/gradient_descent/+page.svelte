<script>
  import Container from "$lib/Container.svelte";
  import Plot from "$lib/Plot.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Mse from "./_loss/Mse.svelte";
  import Slider from "$lib/Slider.svelte";
  import StepButton from "$lib/button/StepButton.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  let parabolaData = [];
  let parabolaWithSlopeData = [];
  let parabolaPoint = [];
  let parabolaStartingPoint = [];

  // parameters for parabola
  let pointX = 55;
  $: pointY = pointX ** 2;
  $: m = 2 * pointX;
  let alpha = 0.01;
  parabolaStartingPoint.push({ x: 55, y: 55 ** 2 });

  //parameters for function with local minimum
  let localPoint = [];
  let localX = 6;
  let localAlpha = 0.01;
  $: localY = localX ** 3 - 5 * localX ** 2 + 10;

  // draw parabola x^2
  for (let i = -60; i <= 60; i++) {
    let x = i;
    let y = x ** 2;
    parabolaData.push({ x, y });
  }

  // draw x^3 - 5x^2 + 10 to show local minimum
  let localMinimumData = [];
  for (let i = -6; i <= 7; i += 0.1) {
    let x = i;
    let y = x ** 3 - 5 * x ** 2 + 10;
    localMinimumData.push({ x, y });
  }

  function recalculatePoints() {
    parabolaPoint = [];
    parabolaPoint.push({ x: pointX, y: pointY });
  }

  function recalculateLocalMinimumPoints() {
    localPoint = [];
    localPoint.push({ x: localX, y: localY });
  }

  function calculateSlope() {
    parabolaWithSlopeData = [[...parabolaData]]; //derivative is 2x
    //the equation is y = mx + t, and we have m (2x) and y+x come from the current point
    //therefore we can find t and draw the line
    let m = 2 * pointX;
    let t = pointY - m * pointX;
    parabolaWithSlopeData.push([
      {
        x: -100,
        y: m * -100 + t,
      },
      {
        x: 100,
        y: m * 100 + t,
      },
    ]);
  }

  function gradientDescentStep() {
    pointX = pointX - alpha * 2 * pointX;
  }

  function localGradientDescent() {
    // we are dealing with x^3 - 5x^2 + 10
    // the derivative is 3x^2 - 10x
    localX = localX - localAlpha * (3 * localX ** 2 - 10 * localX);
  }

  $: pointX && recalculatePoints();
  $: pointX && calculateSlope();
  $: localY && recalculateLocalMinimumPoints();

  // gradient descent mse
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

  function calculateMse() {
    mse = 0;
    dataMse[0].forEach((sample) => {
      mse += (w * sample.x + b - sample.y) ** 2;
    });
    mse /= numSamples;
  }
  calculateMse();

  function mseGradientDescentStep() {
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

  let runs = 0;
  function train() {
    mseGradientDescentStep();
    runs++;
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

  <h2>The Mechanics of Gradient Descent</h2>
  <p>
    Before we discuss how we can use gradient descent to find the optimal
    weights and the optimal bias for linear regression, let us take a step back
    and consider how we can find the value for <Latex>x</Latex> that minimizes
    <Latex>f(x) = x^2</Latex>.
  </p>

  <p>
    The equation <Latex>f(x) = x^2</Latex> depicts a parabola. From visual inspection
    we can determine, that the <Latex>x</Latex> value of 0 produces the minimum <Latex
      >f(x)</Latex
    >.
  </p>
  <Plot
    pathsData={parabolaData}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -80,
      maxX: 80,
      minY: 0,
      maxY: 3000,
      xLabel: "x",
      yLabel: "f(x)",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      numTicks: 5,
    }}
  />
  <p>
    In machine learning we rarely have the luxury of being able to visually find
    the optimal solution. Our functions are usually dependend on thousand or
    million of features and that is not something that we can visualize. We need
    to apply an algorithmic procedure, that finds the minimum automatically. We
    start the algorithm by assigning <Latex>x</Latex> a random value. In the example
    below we picked 55.
  </p>
  <Plot
    pathsData={parabolaData}
    pointsData={parabolaStartingPoint}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -80,
      maxX: 80,
      minY: 0,
      maxY: 3000,
      xLabel: "x",
      yLabel: "f(x)",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
      numTicks: 5,
    }}
  />
  <p>
    Next we calculate the derivative of <Latex>f(x)</Latex> with respect to <Latex
      >x</Latex
    >. Using the rules of basic calculus we derive <Latex
      >{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x) = 2x`}</Latex
    >. The slope at our starting point is therefore 110. We can draw the tangent
    line at the starting point to visualize the derivative.
  </p>

  <Plot
    pathsData={parabolaWithSlopeData}
    pointsData={parabolaStartingPoint}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -80,
      maxX: 80,
      minY: 0,
      maxY: 3000,
      xLabel: "x",
      yLabel: "f(x)",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
      numTicks: 5,
    }}
  />

  <p>
    The derivative shows us the direction of steepest descent and ascent. We can
    use that direction to determine how we can shift <Latex>x</Latex>. When we
    want to minimize <Latex>f(x)</Latex> we use gradient descent and subtract
    <Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex> from <Latex
      >x</Latex
    >.
  </p>
  <p>
    At each timestep we update the variable <Latex>x</Latex> by subtracting the derivative
    until we get close to the minimum, 0 in our case. While the derivative gives
    us the direction in which should take a step, the derivative does not give us
    the size of the step. For that purpose we use a variable <Latex
      >\alpha</Latex
    > also called the
    <Highlight>learning rate</Highlight>. The learning rate scales the
    derivative by multiplying the direction with a value that usually lies
    between 0.1 and 0.001. Larger values of the learning rate could make the
    algorithm diverge. That would mean that the <Latex>f(x)</Latex> would get larger
    and larger and never get close to the minimum.
  </p>
  <Latex
    >{String.raw`x_{t+1} \coloneqq x_t - \alpha \dfrac{\mathrm{d}}{\mathrm{d}x}f(x_t)`}</Latex
  >
  <p>
    Below you can utilize an interactive example to get some intuition for
    gradient descent. Each click on the play button takes a single gradient
    descent step, based on the parameters that you can change with the sliders.
  </p>

  <StepButton on:click={gradientDescentStep} />
  <Plot
    pathsData={parabolaWithSlopeData}
    pointsData={parabolaPoint}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -80,
      maxX: 80,
      minY: 0,
      maxY: 3000,
      xLabel: "x",
      yLabel: "f(x)",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
      numTicks: 5,
    }}
  />
  <div class="flex-container solo light-blue">
    <Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex>
    <p>{m.toFixed(2)}</p>
  </div>
  <div class="flex-container">
    <div class="flex-container left">
      <Latex>\alpha</Latex>
      <p>{alpha}</p>
    </div>
    <Slider min={0.01} max={1.05} bind:value={alpha} step={0.01} />
  </div>
  <div class="flex-container">
    <div class="flex-container left">
      <Latex>x</Latex>
      <p>{pointX.toFixed(2)}</p>
    </div>
    <Slider min={-60} max={60} bind:value={pointX} step={0.1} />
  </div>

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
    Picking the correct learning rate is an extremely usefull skill and is generally
    on of the first things to tweak when you want your algorithm to perform better.
    In fact <Latex>\alpha</Latex> is one of the so called <Highlight
      >hyperparameters</Highlight
    >. A hyperparamter is a parameter that is set by the programmer that
    influences the learning of the parameters that you are truly interested in
    (like <Latex>w</Latex> and <Latex>b</Latex>). A third thing that you should
    notice is the decrease of the derivative when we start getting closer and
    closer to the optimal value. You can also observe that the slope of the
    tangent gets flatter and flatter. This natural behaviour makes sure that we
    take smaller and smaller steps as we start approaching the optimum. This
    also means that gradient descent does not find an optimal value for <Latex
      >x</Latex
    > but an approximative one. In many cases it is sufficient to be close enought
    to the optimal value. At this point the question might arise: "how do we know
    when to stop if we never hit the actual optimal value?". A common approach in
    machine learning is to take a predetermined number of steps. We will discuss
    the topic in more detail throughout this book.
  </p>
  <p>
    While the gradient descent algorithm is the de facto standard in deep
    learning, it has some limitations. Only when we are dealing with a <Highlight
      >convex</Highlight
    > function, we have a guarantee that the algorithm will converge to the global
    optimum. A convex function is like the parabola above, a function that is shaped
    like a "bowl". Such a "bowl" shaped function allows the variable to move towards
    the minimum without any barriers.
  </p>
  <p>
    Below is the graph for the function <Latex>f(x) = x^3 - 5x^2 + 10</Latex>, a
    non convex function. We start at the <Latex>x</Latex> position of 6. If you apply
    gradient several times (play button) you will notice that the ball gets stuck
    in the local minimum and will thus never keep going into the direction of the
    global minimum. This is due to the fact, that at that local minimum point the
    derivative corresponds to 0 and the gradient descent algorithm breaks down. You
    could move the slider below the graph and place the ball to the left of 0 and
    observe that the ball will keep going and going further down.
  </p>
  <StepButton on:click={localGradientDescent} />
  <Plot
    pathsData={localMinimumData}
    pointsData={localPoint}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -3,
      maxX: 6,
      minY: -40,
      maxY: 50,
      xLabel: "x",
      yLabel: "f(x)",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
      xTicks: [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
      yTicks: [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
    }}
  />
  <div class="flex-container">
    <div class="flex-container left">
      <Latex>x</Latex>
      <p>{localX.toFixed(2)}</p>
    </div>
    <Slider min={-3} max={6} bind:value={localX} step={0.1} />
  </div>
  <p>
    This behaviour has several implications that we should discuss. First, the
    starting position of the variable matters and might have an impact on the
    performance. Second, why do deep learning researchers and practicioners use
    gradient descent, if the neural network function is not convex and there is
    a chance that the algorithm will get stuck in a local minimum? Simply put,
    because it works exceptionally well in practice. Additionally, we rarely use
    the "traditional" gradient descent algorithm in practice. Over time,
    researchers discovered that the algorithm can be improved by such ideas as
    "momentum", which keeps the speed of gradient descent over many iterations
    and might thus jump over the local minimum. We will cover those ideas later,
    for now lets focus on the basic algorithm.
  </p>
  <p>
    Before we move on to the part where we discuss how we can apply this
    algorithm to linear regression, let us discuss how we can deal with
    functions that have more than one variable, for example <Latex
      >{String.raw`f(x_1, x_2) = x_1^2 + x_2^2`}</Latex
    >. The approach is actually very similar. Instead of calculating the
    derivative with respect to <Latex>x</Latex>
    <Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex> we need to calculate
    the partial derivatives with respect to all variables, in our case
    <Latex>{String.raw`\dfrac{\mathrm{\partial}}{\mathrm{\partial}x_1}f`}</Latex
    >
    and <Latex
      >{String.raw`\dfrac{\mathrm{\partial}}{\mathrm{\partial}x_2}f`}</Latex
    >. For convenience we put the partial derivatives and the variables into
    their corresponding vectors.
  </p>
  <Latex>
    {String.raw`
\mathbf{x} = 
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
  `},
  </Latex>
  <Latex>
    {String.raw`
\mathbf{\nabla} = 
\begin{bmatrix}
\dfrac{\mathrm{\partial}}{\mathrm{\partial}x_1}f \\[8pt] 
\dfrac{\mathrm{\partial}}{\mathrm{\partial}x_2}f
\end{bmatrix}
  `}
  </Latex>
  <p>
    The gradient descent algorithm looks almost the same. The only difference is
    the substitution of scalars for vectors.
  </p>
  <Latex
    >{String.raw`\mathbf{x}_{t+1} \coloneqq \mathbf{x}_t - \alpha \mathbf{\nabla} `}</Latex
  >
  <p>
    The vector that is represented by
    <Latex>\nabla</Latex> (pronounced <em>nabla</em>) is called the <Highlight
      >gradient</Highlight
    >, giving its name to the gradient descent algorithm.
  </p>
  <div class="separator" />

  <h2>Mean Squared Error Gradient Descent</h2>
  <h3>Single Training Sample</h3>
  <p>
    Let us remind ourselves that our goal is to minimize the mean squared error
    <Latex
      >{String.raw`MSE=\frac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)})^2`}</Latex
    >. In the definition of <Latex>MSE</Latex> we use the upperscript notation <Latex
      >{String.raw`(i)`}</Latex
    > to indicate that there are <Latex>n</Latex> samples in the dataset and <Latex
      >i</Latex
    > is the index of a particular sample. Yet before we move on to discussing how
    we can use gradient descent with multiple data points, let us ease into the calculations
    by assuming that we have one single sample <Latex>n=1</Latex>. That reduces
    the mean squared error to a much simpler form
    <Latex>{String.raw` MSE=(y - \hat{y})^2`}</Latex>. Generally speaking we
    want to find the weight vector <Latex>{String.raw`\mathbf{w}`}</Latex> and the
    bias scalar <Latex>b</Latex> that minimize the mean squared error.
  </p>
  <Latex
    >{String.raw`\underset{\mathbf{w}, b} {\arg\min}(y - (\mathbf{xw^T} + b))^2`}</Latex
  >
  <p>
    Just as in the examples above we will use gradient descent to find the
    minimim of the mean squared error. Yet there is a significant difference in
    the notation. What we need to optimize are the weights <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and the bias<Latex>{String.raw`b`}</Latex> and not the inputs <Latex
      >{String.raw`\mathbf{x}`}</Latex
    >. The <Latex>{String.raw`\mathbf{x}`}</Latex> vector contains the fixed features
    of a single sample.
  </p>
  <p>
    The computation of the gradient is slighly more complicated than the one we
    covered above, because we have to apply the chain rule. To simplify notation
    let us define the expression <Latex>z</Latex> as
    <Latex>{String.raw`z = y - (\mathbf{xw^T} + b)`}</Latex>. That way we can
    define <Latex>MSE</Latex> as <Latex>{String.raw`z^2`}</Latex>.
  </p>

  <p>
    In order to be able to apply gradient descent we need to calculate partial
    derivatives with respect to each weight <Latex>w_j</Latex> and the bias <Latex
      >b</Latex
    >. We will not calculate the derivatives directly, but apply the chain rule
    by utilizing <Latex>z</Latex>.
  </p>
  <Latex
    >{String.raw`\dfrac{\partial MSE}{\partial w_j} = \dfrac{\partial MSE}{\partial z} \dfrac{\partial z}{\partial w_j}`}</Latex
  >,
  <br />
  <Latex
    >{String.raw`\dfrac{\partial MSE}{\partial b} = \dfrac{\partial MSE}{\partial z} \dfrac{\partial z}{\partial b}`}</Latex
  >
  <p>
    Using basic rules of calculus we derive the following partial derivatives.
  </p>
  <Latex>{String.raw`\dfrac{\partial MSE}{\partial z} = 2z`}</Latex>,
  <br />
  <Latex>{String.raw`\dfrac{\partial z}{\partial w_j} = -x_j`}</Latex>,
  <br />
  <Latex>{String.raw`\dfrac{\partial z}{\partial b} = -1`}</Latex>
  <p>
    By applying the chain rule we end up with the desired partial derivatives.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
  \dfrac{\partial MSE}{\partial w_j} & = \dfrac{\partial MSE}{\partial z} \dfrac{\partial z}{\partial w_j} \\
& = -2zx_j \\
& = -2x_j (y - (\mathbf{xw^T} + b))
  \end{aligned}
  `}</Latex
  >
  <br />
  <Latex
    >{String.raw`
  \begin{aligned}
  \dfrac{\partial MSE}{\partial b} & = \dfrac{\partial MSE}{\partial z} \dfrac{\partial z}{\partial b} \\
& = -2z\\
& = -2 (y - (\mathbf{xw^T} + b))
  \end{aligned}
  `}</Latex
  >
  <p>
    Often you will see a slighly different definition of the mean squared error,
    where the MSE is divided by 2.
  </p>
  <Latex
    >{String.raw`MSE=\dfrac{1}{2n}\sum_i^n (y^{(i)} - \hat{y}^{(i)})^2`}</Latex
  >
  <p>
    This is done for convenience. If you look at the partial derivatives above
    you will notice, that all of them contain the number 2. The division by 2
    therefore cancels the 2 in the derivative and makes the results more
    compact.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
  \dfrac{\partial MSE}{\partial w_j} & = \dfrac{\partial MSE}{\partial z} \dfrac{\partial z}{\partial w_j} \\
& = -\dfrac{1}{2}2z * x_j \\
& = -x_j (y - (\mathbf{xw^T} + b))
  \end{aligned}
  `}</Latex
  >
  <br />
  <Latex
    >{String.raw`
  \begin{aligned}
  \dfrac{\partial MSE}{\partial b} & = \dfrac{\partial MSE}{\partial z} \dfrac{\partial z}{\partial b} \\
& = -\dfrac{1}{2}2z \\
& =  -(y - (\mathbf{xw^T} + b))
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
    >{String.raw`\mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w `}</Latex
  >
  <br />
  <Latex
    >{String.raw`b_{t+1} \coloneqq b_t - \alpha \dfrac{\partial}{\partial b} `}</Latex
  >
  <p>
    While didactically it makes sense to learn how we can calculate the gradient
    for a single sample first, practically we always deal with much larger
    datasets, often consiting of many thousands or even millions of samples. In
    practice we use <Highlight>batch</Highlight>, <Highlight
      >stochastic</Highlight
    > or <Highlight>mini-batch</Highlight> gradient descent.
  </p>
  <h3>Batch Gradient Descent</h3>
  <p>
    As it turns out making a jump from one to several samples is not that
    complicated. You should remember from calculus that the derivative of a sum
    is the sum of derivatives. In other words in order to calculate the gradient
    of the mean squared error, we need to calculate the individual gradients for
    each sample and calculate the mean.
  </p>
  <Latex
    >{String.raw`
\begin{aligned}
\large 
& \mathbf{\nabla}_{w} = \dfrac{1}{n} \sum_i^n\mathbf{\nabla}^{(i)}_w \\
& \dfrac{\partial}{\partial b} = \dfrac{1}{n}\sum^n_i\dfrac{\partial}{\partial b}^{(i)}
\end{aligned}
    `}</Latex
  >
  <p>
    The approach of using the whole dataset to calculate the gradient is called <Highlight
      >batch</Highlight
    > gradient descent. Using the whole dataset has the advantage that we get a good
    estimation for the gradients, yet in many cases batch gradient descent is not
    used in practice. We often have to deal with datasets consisting of thousands
    of features and millions of samples. It is not possible to load all that data
    on the GPU's. Even if it was possible, it would take a lot of time to calculate
    the gradients for all the samples in order to take just a single training step.
    The alternatives described below are more practical and usually converge a lot
    faster.
  </p>

  <p>
    Below is the interactive example from the last section. This time we use
    batch gradient descent to find the optimal weight and bias. As we have only
    4 datapoints, batch gradient descent is a fine choice.
  </p>

  <PlayButton f={train} delta={1} />
  <Mse data={dataMse} {w} {b} />

  <h3>Stochastic Gradient Descent</h3>
  <p>
    In stochastic gradient descent we introduce some stochasticity by shuffling
    the dataset randomly and using one sample at a time to calculate the
    gradient and to take a gradient descent step until we have used all samples
    in the dataset. This period of time, in which we exhaust all samples in the
    training dataset is called an <Highlight>epoch</Highlight>. After each epoch
    we reshuffle the data and start over. The advantage of stochastic gradient
    descent is that we do not have to wait for the calculation of gradients for
    all samples, but in the process we lose the advantages of parallelization
    that we get with batch gradient descent.
  </p>
  <p>
    When we calculate the gradient based on one sample the calculation is going
    to be off. By iterating over the whole dataset the sum of the directions is
    going to move the weights and biases towards the optimum. In fact this
    behaviour is often seen as advantageous, because theoretically the imprecise
    gradient could potentially push a variable from a local minimum.
  </p>

  <h3>Mini-Batch Gradient Descent</h3>
  <p>
    Mini-batch gradient descent combines the advantages of the stochastic and
    batch gradient descent. At the start of each epoch the dataset is shuffled
    randomly, but insdead of using one sample at a time in mini-batch gradient
    descent several samples are taken. Similar to the learning rate the size of
    the mini-batch is a hyperparameter and needs to be determined by the
    developer. Usually the size is calculated as a power of 2, for example 32,
    64, 128 and so on. You just need to remember that the batch needs to fit
    into the memory of your graphics card.
  </p>
  <p>
    Mini-batch gradient descent can be parallelized, because we use several
    samples at a time. Additionally it has the advantage that theoretically our
    training dataset can be as large as we want it to be.
  </p>
  <div class="separator" />
  <h2>The Case for Gradient Descent</h2>
  <p>
    If you have taken a statistics course, you might remember, that there is an
    explicit solution to the linear regression problem, which does not involve
    gradient descent.
  </p>
  <Latex
    >{String.raw`\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}`}</Latex
  >
  <p>
    While it is true that we could use that equation to find the exact weights
    and biases that minimize the mean squared error, this approach does not
    scale well. If you look at the equation, you will notice that the
    calculation of the inverse of a matrix is required, which would slow down
    the calculation significantly as the number of training samples grows. In
    deep learning, where millions and millions of samples are required, this is
    not a feasible solution.
  </p>
  <p>
    Even if computation was not a major bottleneck, neural networks do not
    provide an explicit solution, therefore we are dependent on gradient
    descent. This algorithm will follow us until the end of this block and even
    beyond.
  </p>
  <div class="separator" />
</Container>

<style>
  .flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .left p {
    margin-left: 10px;
  }
  .left {
    min-width: 110px;
    margin-right: 10px;
  }

  .solo p {
    padding: 0px 10px;
  }
</style>
