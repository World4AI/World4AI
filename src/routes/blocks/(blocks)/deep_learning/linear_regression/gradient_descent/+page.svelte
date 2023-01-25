<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Slider from "$lib/Slider.svelte";
  import Alert from "$lib/Alert.svelte";

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
</script>

<svelte:head>
  <title>Gradient Descent - World4AI</title>
  <meta
    name="description"
    content="Gradient descent is the algorithm that utilizes caclulus to find the parameters, that minimize the function. The algorithm works in an iterative fashion, where we change the parameters until a certain stop criterion is met."
  />
</svelte:head>

<Container>
  <h1>Gradient Descent</h1>
  <div class="separator" />

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
</Container>
