<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Slider from "$lib/Slider.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Mse from "../_loss/Mse.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Legend from "$lib/plt/Legend.svelte";

  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import StepButton from "$lib/button/StepButton.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  let parabolaData = [];
  let parabolaPoint = [];
  let parabolaStartingPoint = [];
  let slope = [];

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
    //the equation is y = mx + t, and we have m (2x) and y+x come from the current point
    //therefore we can find t and draw the line
    slope = [];
    let m = 2 * pointX;
    let t = pointY - m * pointX;
    slope.push({ x: -100, y: m * -100 + t });
    slope.push({ x: 100, y: m * 100 + t });
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
    { x: 5, y: 20 },
    { x: 10, y: 40 },
    { x: 35, y: 15 },
    { x: 45, y: 59 },
  ];

  let w = 1;
  let b = 1;
  let numSamples = dataMse.length;
  let mse;
  let mseAlpha = 0.001;

  function calculateMse() {
    mse = 0;
    dataMse.forEach((sample) => {
      mse += (w * sample.x + b - sample.y) ** 2;
    });
    mse /= numSamples;
  }
  calculateMse();

  function mseGradientDescentStep() {
    let db = 0;
    let dw = 0;

    dataMse.forEach((sample) => {
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
  <title>Linear Regression Gradient Descent - World4AI</title>
  <meta
    name="description"
    content="Gradient descent is the algorithm that is most commonly used to find the optimal weights and biases in linear regression. In this section we discuss the basics of gradiennt descent and apply the algorithm to linear regression."
  />
</svelte:head>

<Container>
  <h1>Gradient Descent</h1>
  <div class="separator" />

  <h2>The Mechanics of Gradient Descent</h2>
  <p>
    Before we discuss how we can find the optimal weights and the optimal bias
    in a linear regression setting, let us take a step back and consider how we
    can find the value of variable <Latex>x</Latex> that minimizes the function <Latex
      >f(x) = x^2</Latex
    >.
  </p>

  <p>
    The equation <Latex>f(x) = x^2</Latex> depicts a parabola. From visual inspection
    we can determine, that the <Latex>f(x)</Latex> is lowest when <Latex
      >x</Latex
    >. is exactly 0.
  </p>

  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-80, 80]}
    range={[0, 3000]}
    padding={{ top: 40, right: 10, bottom: 40, left: 60 }}
  >
    <Ticks
      xTicks={[-80, -60, -40, -20, 0, 20, 40, 60, 80]}
      yTicks={[0, 500, 1000, 1500, 2000, 2500, 3000]}
      xOffset={-10}
      yOffset={25}
    />
    <XLabel text="x" fontSize={15} type="latex" />
    <YLabel text="f(x)" fontSize={15} type="latex" x={-3} />
    <Path data={parabolaData} />
    <Circle data={[{ x: 0, y: 0 }]} />
  </Plot>

  <p>
    In machine learning we rarely have the luxury of being able to visually find
    the optimal solution. Our function is usually dependend on thousands or
    millions of features and that is not something that we can visualize. We
    need to apply an algorithmic procedure, that finds the minimum
    automatically. We start the algorithm by assigning <Latex>x</Latex> a random
    value. In the example below we picked 55.
  </p>

  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-80, 80]}
    range={[0, 3000]}
    padding={{ top: 40, right: 10, bottom: 40, left: 60 }}
  >
    <Ticks
      xTicks={[-80, -60, -40, -20, 0, 20, 40, 60, 80]}
      yTicks={[0, 500, 1000, 1500, 2000, 2500, 3000]}
      xOffset={-10}
      yOffset={25}
    />
    <XLabel text="x" fontSize={15} type="latex" />
    <YLabel text="f(x)" fontSize={15} type="latex" x={-3} />
    <Path data={parabolaData} />
    <Circle data={parabolaStartingPoint} />
  </Plot>

  <p>
    Next we calculate the derivative of <Latex>f(x)</Latex> with respect to <Latex
      >x</Latex
    >. Using the rules of basic calculus we derive <Latex
      >{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x) = 2x`}</Latex
    >. The slope at our starting point is therefore 110. We can draw the tangent
    line at the starting point to visualize the derivative.
  </p>

  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-80, 80]}
    range={[0, 3000]}
    padding={{ top: 40, right: 10, bottom: 40, left: 60 }}
  >
    <Ticks
      xTicks={[-80, -60, -40, -20, 0, 20, 40, 60, 80]}
      yTicks={[0, 500, 1000, 1500, 2000, 2500, 3000]}
      xOffset={-10}
      yOffset={25}
    />
    <XLabel text="x" fontSize={15} type="latex" />
    <YLabel text="f(x)" fontSize={15} type="latex" x={-3} />
    <Path data={parabolaData} />
    <Path data={slope} />
    <Circle data={parabolaStartingPoint} />
  </Plot>
  <p>
    The derivative shows us the direction of steepest descent, or simply put the
    derivative tells us in what direction we have to change <Latex>x</Latex>
    if we want to reduce <Latex>f(x)</Latex>. The <Highlight
      >gradient descent</Highlight
    > algorithm utilizes that directions and simply subtract the derivative
    <Latex>{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex> from <Latex
      >x</Latex
    >. Gradient descent is an iterative algorithm. That means that we keep
    calculating the derivative <Latex
      >{String.raw`\dfrac{\mathrm{d}}{\mathrm{d}x}f(x)`}</Latex
    > and updating the variable <Latex>x</Latex> until some criterion is met. For
    example once the change in <Latex>x</Latex> is below a certain threshhold, we
    can assume that we are very close to the minimum.
  </p>
  <p>
    While the derivative gives us the direction in which should take a step, the
    derivative does not give us the size of the step. For that purpose we use a
    variable <Latex>\alpha</Latex>, also called the
    <Highlight>learning rate</Highlight>. The learning rate scales the
    derivative by multiplying the direction with a value that usually lies
    between 0.1 and 0.001. Larger values of the learning rate could make the
    algorithm diverge. That would mean that <Latex>f(x)</Latex> would get larger
    and larger and never get close to the minimum. While too low values would slow
    down the trainig process dramatically.
  </p>
  <Alert type="info">
    <p>
      At each time step <Latex>t</Latex> of the gradient descent algorithm we update
      the variable <Latex>x</Latex>, until <Latex>f(x)</Latex> converges to the miminum.
    </p>
    <Latex
      >{String.raw`x_{t+1} \coloneqq x_t - \alpha \dfrac{\mathrm{d}}{\mathrm{d}x}f(x_t)`}</Latex
    >
  </Alert>
  <p>
    Below you can play with an interactive example to get some intuition
    regarding the gradient descent algorithm. Each click on the play button
    takes a single gradient descent step, based on the parameters that you can
    change with the sliders.
  </p>

  <ButtonContainer>
    <StepButton on:click={gradientDescentStep} />
  </ButtonContainer>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-80, 80]}
    range={[0, 3000]}
    padding={{ top: 40, right: 10, bottom: 40, left: 60 }}
  >
    <Ticks
      xTicks={[-80, -60, -40, -20, 0, 20, 40, 60, 80]}
      yTicks={[0, 500, 1000, 1500, 2000, 2500, 3000]}
      xOffset={-10}
      yOffset={25}
    />
    <XLabel text="x" fontSize={15} type="latex" />
    <YLabel text="f(x)" fontSize={15} type="latex" x={-3} />
    <Path data={parabolaData} />
    <Path data={slope} />
    <Legend
      text="Derivative: {m.toFixed(2)}"
      coordinates={{ x: -30, y: 2800 }}
      fontSize={20}
    />
    <Circle data={parabolaPoint} />
  </Plot>
  <div class="flex justify-center items-center">
    <div class="flex justify-center items-center w-28">
      <Latex>\alpha</Latex>
      <p class="m-0 ml-2">{alpha}</p>
    </div>
    <Slider min={0.01} max={1.05} bind:value={alpha} step={0.01} />
  </div>
  <div class="flex justify-center items-center">
    <div class="flex justify-center items-center w-28">
      <Latex>x</Latex>
      <p class="m-0 ml-2">{pointX.toFixed(2)}</p>
    </div>
    <Slider min={-60} max={60} bind:value={pointX} step={0.1} />
  </div>

  <p>
    You can learn several things about gradient descent if you play with the
    example.
  </p>
  <ol class="list-decimal list-inside">
    <li class="mb-2">
      If you try positive and negative <Latex>x</Latex> values you will observe that
      the sign of the derivative changes based on the sign of the location of <Latex
        >x</Latex
      >. That behaviour makes sure that we distract negative values from <Latex
        >x</Latex
      > when <Latex>x</Latex> is negative and we distract positive values from <Latex
        >x</Latex
      > when <Latex>x</Latex> is positive. No matter where we start, the algorithm
      always pushes the variable towards the minimum.
    </li>
    <li class="mb-2">
      If you try gradient descent with an <Latex>\alpha</Latex> of 1.01 you will
      observe that the algorithm starts to diverge. Picking the correct learning
      rate is an extremely usefull skill and is generally on of the first things
      to tweak when you want your algorithm to perform better. In fact <Latex
        >\alpha</Latex
      > is one of the so called
      <Highlight>hyperparameters</Highlight>. A hyperparamter is a parameter
      that is set by the programmer that influences the learning of the
      parameters that you are truly interested in (like <Latex>w</Latex> and <Latex
        >b</Latex
      >).
    </li>
    <li class="mb-2">
      You should also notice the decrease of the magnitude of the derivative
      when we start getting closer and closer to the optimal value, whiel the
      slope of the tangent gets flatter and flatter. This natural behaviour
      makes sure that we take smaller and smaller steps as we start approaching
      the optimum. This also means that gradient descent does not find an
      optimal value for <Latex>x</Latex> but an approximative one. In many cases
      it is sufficient to be close enough to the optimal value.
    </li>
  </ol>
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
    non convex function. We start at the <Latex>x</Latex> position with a value of
    6. If you apply gradient several times (arrow button) you will notice that the
    ball gets stuck in the local minimum and will thus never keep going into the
    direction of the global minimum. This is due to the fact, that at that local
    minimum point the derivative corresponds to 0 and the gradient descent algorithm
    breaks down. You could move the slider below the graph and place the ball to
    the left of 0 and observe that the ball will keep going and going further down.
  </p>
  <ButtonContainer>
    <StepButton on:click={localGradientDescent} />
  </ButtonContainer>
  <Plot
    width={500}
    height={250}
    maxWidth={600}
    domain={[-3, 6]}
    range={[-40, 50]}
    padding={{ top: 40, right: 10, bottom: 40, left: 50 }}
  >
    <Ticks
      xTicks={[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]}
      yTicks={[-40, -30, -20, -10, 0, 10, 20, 30, 40, 50]}
      xOffset={-15}
      yOffset={25}
    />
    <XLabel text="x" fontSize={15} type="latex" />
    <YLabel text="f(x)" fontSize={15} type="latex" x={0} />
    <Path data={localMinimumData} />
    <Circle data={localPoint} />
  </Plot>
  <div class="flex justify-center items-center">
    <div class="flex justify-center items-center w-24">
      <Latex>x</Latex>
      <p class="m-0 ml-2">{localX.toFixed(2)}</p>
    </div>
    <Slider min={-3} max={6} bind:value={localX} step={0.1} />
  </div>
  <p>
    This behaviour has several implications that we should discuss. First, the
    starting position of the variable matters and might have an impact on the
    performance. Second, the following question arises: "why do deep learning
    researchers and practicioners use gradient descent, if the neural network
    function is not convex and there is a chance that the algorithm will get
    stuck in a local minimum?". Simply put, because it works exceptionally well
    in practice. Additionally, we rarely use the "traditional" gradient descent
    algorithm in practice. Over time, researchers discovered that the algorithm
    can be improved by such ideas as "momentum", which keeps the speed of
    gradient descent over many iterations and might thus jump over the local
    minimum. We will cover those ideas later, for now lets focus on the basic
    algorithm.
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
      >{String.raw`MSE=\dfrac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)})^2`}</Latex
    >. We use the upperscript notation <Latex>{String.raw`(i)`}</Latex> to indicate
    that there are <Latex>n</Latex> samples in the dataset and <Latex>i</Latex> is
    the index of a particular sample. To make our journey easier, let us for now
    assume that we have one single sample. That reduces the mean squared error to
    a much simpler form
    <Latex>{String.raw` MSE=(y - \hat{y})^2`}</Latex>.
  </p>
  <p>
    Generally speaking we want to find the weight vector <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and the bias scalar <Latex>b</Latex> that minimize the mean squared error using
    gradient descent.
  </p>
  <Latex
    >{String.raw`\underset{\mathbf{w}, b} {\arg\min}(y - (\mathbf{xw^T} + b))^2`}</Latex
  >
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
    >. Using the chain rule we get the following derivatives.
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
  <p>
    Once we have the gradients, the gradient descent algorithm works as
    expected.
  </p>
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
    If we wanted to implement batch gradient descent with Python and NumPy to
    find the minimum squared error, we could implement the algorithm in just 6
    lines of code.
  </p>
  <PythonCode
    code={`for epoch in range(epochs):
    # 1. calculate output of linear regression
    y_hat = X @ w.T + b
    # 2. calculate the gradients 
    grad_w = (-X * (y - y_hat)).mean(axis=0) 
    grad_b = -(y - y_hat).mean()
    # 3. apply batch gradient descent
    w = w - alpha * grad_w
    b = b - alpha * grad_b`}
  />
  <p>
    Below you can use the example from the last section to get a visual
    intuition for how the algorithm works.
  </p>

  <ButtonContainer>
    <PlayButton f={train} delta={1} />
  </ButtonContainer>
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
    descent.
  </p>
  <div class="separator" />
</Container>
