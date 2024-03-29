<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Slider from "$lib/Slider.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";

  let linearData = [];
  let b = 0;
  let w = 5;
  for (let i = -100; i < 100; i++) {
    let x = i;
    let y = b + w * (x + Math.random() * 20 - 10);
    linearData.push({ x, y });
  }

  let nonlinearData = [];
  for (let i = -100; i < 100; i++) {
    let x = i + 1;
    let y = (x + Math.random() * 10 - 10) ** 2;
    nonlinearData.push({ x, y });
  }

  let line = [
    { x: -100, y: 0 },
    { x: 100, y: 0 },
  ];

  let estimatedBias = -200;
  let estimatedWeight = -100;

  function calculatePoints(estimatedBias, estimatedWeight) {
    let y1 = estimatedBias + estimatedWeight * line[0].x;
    let y2 = estimatedBias + estimatedWeight * line[1].x;
    line[0].y = y1;
    line[1].y = y2;
    line = line;
  }

  $: calculatePoints(estimatedBias, estimatedWeight);
</script>

<svelte:head>
  <title>Linear Model - World4AI</title>
  <meta
    name="description"
    content="A linear model allows us to model the data using a line (or a hyperplane) in the coordinate system."
  />
</svelte:head>

<Container>
  <h1>Linear Model</h1>
  <div class="separator" />
  <p>
    The term "linear regression" consists of two words, that fully describe the
    type of model we are dealing with: <Highlight>linear</Highlight> and <Highlight
      >regression</Highlight
    >. The "regression" part signifies that our model predicts a numeric target
    variable based on given features and we are not dealing with a
    classification task. The "linear" part suggests that linear regression can
    only model a linear relationship between features and targets. To clarify
    what the words "linear relationship" mean we present two examples below.
  </p>

  <p>
    In the first scatterplot we could plot a line that goes from the coordinates
    of (-100, -500) to coordinates (100, 500). While there is some randomness in
    the data, the line would depict the relationship between the feature and the
    target relatively well. When we get new data points we can use the line to
    predict the target and be relatively confident regarding the outcome.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-100, 100]}
    range={[-500, 500]}
    padding={{ top: 40, right: 15, bottom: 65, left: 65 }}
  >
    <Ticks
      xTicks={[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]}
      yTicks={[-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]}
      fontSize={18}
      xOffset={-25}
      yOffset={30}
    />
    <XLabel text="Feature" fontSize={30} x={280} />
    <YLabel text="Target" fontSize={30} x={15} />
    <Path
      data={[
        { x: -100, y: -500 },
        { x: 100, y: 500 },
      ]}
    />
    <Circle data={linearData} radius={3} />
  </Plot>

  <p>
    In contrast the data in the following scatterplot represents a nonlinear
    relationship between the feature and the target. Theoretically there is
    nothing that stops us from using linear regression for the below problem,
    but there are better alternatives (like neural networks) for non linear
    problems.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-100, 100]}
    range={[0, 10000]}
    padding={{ top: 40, right: 14, bottom: 65, left: 100 }}
  >
    <Ticks
      xTicks={[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]}
      yTicks={[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]}
      fontSize={18}
      xOffset={-25}
      yOffset={45}
    />
    <XLabel text="Feature" fontSize={30} />
    <YLabel text="Target" fontSize={30} x={15} />
    <Circle data={nonlinearData} radius={3} />
    <Path
      data={[
        { x: -100, y: 10000 },
        { x: 100, y: 0 },
      ]}
    />
  </Plot>
  <p>
    From basic math we know, that in the two dimensional space we can draw a
    line using the equation <Latex>y = xw + b</Latex>, where <Latex>x</Latex> is
    the only feature, <Latex>y</Latex> is the target, <Latex>w</Latex> is the weight
    that we use to scale the feature and <Latex>b</Latex> is the bias. While we can
    easily understand that the feature <Latex>x</Latex> is the input of our equation
    and the label <Latex>y</Latex> is the output of the equation, we have a harder
    time imagining what role the weight <Latex>w</Latex> and the bias <Latex
      >b</Latex
    > play in the equation. Below we present two possible interpretations.
  </p>
  <p>
    When we look at the equation <Latex>y = xw + b</Latex> from the arithmetic perspective,
    we should notice two things. First, the output <Latex>y</Latex> equals the bias
    when the input <Latex>x</Latex> is 0: <Latex>y = 0w + b</Latex>. The bias in
    a way encompasses a starting point for the calculation of the output. If for
    example we tried to model the relationship between age and height, even at
    birth (age 0) a human would have some average height, which would be encoded
    in the bias <Latex>b</Latex>. Second, for each unit of <Latex>x</Latex>, the
    output increases by exactly <Latex>w</Latex>. The equation <Latex
      >y = x*5cm + 50cm</Latex
    > would indicate that on average a human grows by 5cm for each year in life.
    At this point you would hopefully interject that this relation is out of touch
    with reality. For once the equation does not reflect that a human being grows
    up to a certain length or that a child grows at a higher rate, than a young adult.
    At a certain age people even start to shrink. While all these points are valid,
    we make specific assumtions, when we model the world using linear regression.
  </p>
  <Alert type="warning">
    When we use a linear regression model, we assume a linear relationship
    between the inputs and the output. If you apply linear regression to data
    that is nonlinear in nature, you might get illogical results.
  </Alert>
  <p>
    When on the other hand we look at the equation <Latex>y = xw + b</Latex> from
    the geometric perspective, we should realize, that weight determines the rotation
    (slope) of the line while the bias determines the horizontal position. Below
    we present an interactive example to demonstrate the impact of the weight and
    the bias on the the regression line. You can move the two sliders to change the
    weight and the bias. Observe what we mean when we say rotation and position.
    Try to position the line, such that it <Highlight>fits</Highlight> the data as
    good as possible.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-100, 100]}
    range={[-500, 500]}
    padding={{ top: 40, right: 15, bottom: 65, left: 65 }}
  >
    <Ticks
      xTicks={[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]}
      yTicks={[-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]}
      fontSize={18}
      xOffset={-25}
      yOffset={30}
    />
    <XLabel text="Feature" fontSize={30} x={280} />
    <YLabel text="Target" fontSize={30} x={15} />
    <Circle data={linearData} radius={3} />
    <Path data={line} stroke={2} />
  </Plot>
  <Slider
    label="Weight"
    labelId="weight"
    showValue={true}
    bind:value={estimatedWeight}
    min={-200}
    max={200}
  />
  <Slider
    label="Bias"
    labelId="bias"
    showValue={true}
    bind:value={estimatedBias}
    min={-500}
    max={500}
  />
  <p>
    We used the weight <Latex>w</Latex> of 5 and the bias <Latex>b</Latex> of 0 plus
    some randomness to generate the data above. When you played with sliders you
    should have come relatively close.
  </p>
  <p>
    The weight and the bias are learnable parameters. The linear regression
    algorithm provides us with a way to find those parameters. You can imagine
    that the algorithm rotates and moves the line, until the line <Highlight
      >fits</Highlight
    > the data. This process is called data or curve fitting.
  </p>
  <p>
    In practice we rarely deal with a dataset where we only have one feature. In
    that case our equation looks as follows.
  </p>
  <Latex>y = x_1w_1 + x_2w_2 + ... + x_mw_m+ b</Latex>
  <p>
    We can also use a more compact form and write the equation in vector form.
  </p>
  <Latex
    >{String.raw`
    y = \mathbf{x} \mathbf{w}^T + b \\
    \mathbf{x} = 
    \begin{bmatrix}
       x_1 & x_2 & \cdots & x_n
    \end{bmatrix} \\
    \mathbf{w} = 
    \begin{bmatrix}
      w_1 & 
      w_2 & 
      \cdots &
      w_m
    \end{bmatrix}
`}</Latex
  >
  <p>
    In a three dimensional space we calculate a two dimensional plane that
    divides the coordinate system into two regions. This procedure is harder to
    imagine for more than 3 dimensions, but we still create a plane (a so called
    hyperplane) in space. The weights are used to rotate the hyperplane while
    the bias moves the plane.
  </p>
  <p>
    When we use linear regression to make predictions based on features, we draw
    a "hat" over the <Latex>y</Latex> value to indicate that we are dealing with
    a prediction from a model,
    <Latex>{String.raw`\hat{y} = \mathbf{x} \mathbf{w}^T + b`}</Latex>. The <Latex
      >y</Latex
    > value on the other hand represents the actual target from the dataset, the
    so called ground truth. Usually we want to create predictions not for a single
    sample <Latex>{String.raw`\mathbf{x}`}</Latex>, but for a whole dataset
    <Latex>{String.raw`\mathbf{X}`}</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{X} =
      \begin{bmatrix}
      x_1^{(1)} & x_2^{(1)} & x_3^{(1)} & \cdots & x_m^{(1)} \\
      x_1^{(2)} & x_2^{(2)} & x_3^{(2)} & \cdots & x_m^{(2)} \\
      x_1^{(3)} & x_2^{(3)} & x_3^{(/3)} & \cdots & x_m^{(3)} \\
      \vdots & \vdots & \vdots & \cdots & \vdots \\
      x_1^{(n)} & x_2^{(n)} & x_3^{(n)} & \cdots & x_m^{(n)} \\
      \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    <Latex>{String.raw`\mathbf{X}`}</Latex> is an <Latex>n \times m</Latex> matrix,
    where <Latex>n</Latex> (rows) is the number of samples and <Latex>m</Latex> (columns)
    is the number of input features. We can multiply the dataset matrix <Latex
      >{String.raw`\mathbf{X}`}</Latex
    > with the transposed weight vector<Latex>{String.raw`\mathbf{w}`}</Latex> and
    add the bias <Latex>b</Latex> to generate a prediction vector <Latex
      >{String.raw`\mathbf{\hat{y}}`}</Latex
    >.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{\hat{y}} = \mathbf{X}\mathbf{w}^T + b
    `}</Latex
    >
  </div>
  <p>
    The advantage of the above procedure is not only due to a more compact
    representation, but has also practical implications. Matrix operations in
    all modern deep learning frameworks can be parallelized. Therefore when you
    utilize matrix notation in your code, you actually make use of that
    parallelism and can speed up your code tremendously. Think about it. Each
    row of the dataset can be multiplied with the weight vector independently.
    By outsourcing the calculations to different CPU or GPU cores, a lot of
    computation time can be saved.
  </p>
  <p>
    By this point you might have noticed, that there is something fishy about
    the expression.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
      \mathbf{\hat{y}} = \mathbf{X}\mathbf{w}^T + b
    `}</Latex
    >
  </div>
  <p>
    On the one side we have a vector that results from <Latex
      >{String.raw`\mathbf{Xw}^T`}</Latex
    >, on the other side we have a scalar <Latex>b</Latex>. From a mathematical
    standpoint adding a scalar to a vector is techincally not allowed. From the
    programming standpoint this procedure is valid, because NumPy and all deep
    leanring frameworks utilize a technique called <Highlight
      >broadcasting</Highlight
    >. We will have a closer look at broadcasting in our practical sessions, for
    now it is sufficient to know, that broadcasting expands scalars, vectors and
    matrices in order for the calculations to make sense. In our example above
    for example, the scalar would be expanded into a vector, which would be of
    the same size as the vector that results from <Latex
      >{String.raw`\mathbf{Xw}^T`}</Latex
    >. We will often include notation that incorporates broadcasting in order to
    make the notation more similar to our Python code.
  </p>
  <p>
    Now let's see how we can impelemnt this idea of a linear model in PyTorch.
  </p>
  <PythonCode
    code={String.raw`import torch
import sklearn.datasets as datasets
`}
  />
  <p>
    We make use of the <code>make_regression()</code> function from the sklearn library
    to make a dataset with 100 samples and 2 features.
  </p>
  <PythonCode
    code={String.raw`X, y = datasets.make_regression(n_samples=100, n_features=2, n_informative=2, noise=0.01)
`}
  />
  <p>
    The above function returns numpy arrays <Latex
      >{String.raw`\mathbf{X}`}</Latex
    > and <Latex>{String.raw`\mathbf{y}`}</Latex> and we transform those into PyTorch
    tensors.
  </p>
  <PythonCode
    code={String.raw`X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)
`}
  />
  <p>
    We initialze the two weights and the bias randomly, using the <code
      >torch.randn()</code
    > function. This function returns random variables, that are drawn from the standard
    normal distribution.
  </p>
  <PythonCode
    code={String.raw`w = torch.randn(1, 2)
b = torch.randn(1, 1)
`}
  />
  <p>The actual model predictions can be calculated using a one liner.</p>
  <PythonCode
    code={String.raw`y_hat = X @ w.T + b
y_hat.shape
`}
  />
  <pre class="text-sm">torch.Size([100, 1])</pre>
  <p>
    While it is relatively easy to use a linear model in PyTorch, we have still
    not encountered any methods to generate predictions that are as close to the
    true labels in the dataset as possible. In the next sections we are going to
    cover how the learning procedure actually works.
  </p>
  <div class="separator" />
</Container>
