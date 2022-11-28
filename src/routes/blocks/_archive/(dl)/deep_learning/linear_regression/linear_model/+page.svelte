<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Slider from "$lib/Slider.svelte";

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
  <title>World4AI | Deep Learning | Linear Model</title>
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
    only model the relationship between features and targets in a linear
    fashion. To clarify what the words "linear relationship" mean we present two
    examples below.
  </p>

  <p>
    In the first scatterplot we could plot a line that goes from the coordinates
    of (-100, -500) and goes to coordinates of (100, 500). While there is some
    randomness in the data, the line would depict the relationship between the
    feature and the target relatively well. When we get new data points we can
    use the line to predict the target and be relatively confident regarding the
    outcome.
  </p>
  <Plot maxWidth={800} domain={[-100, 100]} range={[-500, 500]}>
    <Ticks xTicks={[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]} yTicks={[-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]} xOffset={-15} yOffset={15}/>
    <XLabel text="Feature" fontSize={15}/>
    <YLabel text="Target" fontSize={15}/>
    <Circle data={linearData} radius={3} />
  </Plot>

  <p>
    In contrast the data in the following scatterplot represents a nonlinear
    relationship between the feature and the target. Theoretically there is
    nothing that stops us from using linear regression for the below problem,
    but there are better alternatives (like neural networks) for non linear
    problems.
  </p>
  <Plot maxWidth={800} domain={[-100, 100]} range={[0, 10000]}>
    <Ticks xTicks={[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]} yTicks={[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]} xOffset={-15} yOffset={15}/>
    <XLabel text="Feature" fontSize={15}/>
    <YLabel text="Target" fontSize={15}/>
    <Circle data={nonlinearData} radius={3} />
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
    with reality. For once the equation does not reflect that a human being growth
    up to a certain length or that a child growth at a higher rate, than a young
    adult. At a certain age people even start to shrink. While all these points are
    valid, the assumtion that we always make, when we model the world using linear
    regression is: there is a linear relationship between the inputs and the output.
    If you apply linear regression to data that is nonlinear in nature, you might
    get illogical results.
  </p>
  <p>
    When on the other hand we look at the equation <Latex>y = xw + b</Latex> from
    the geometric perspective, we should realize, that weight determines the rotation
    (slope) of the line while the bias determines the horizontal position. Below
    we present an interactive example to demonstrate the impact of the weight and
    the bias on the form of the line. You can move the two sliders to change the
    weight and the bias. Observe what we mean when we say rotation and position.
    Try to position the line, such that it <Highlight>fits</Highlight> the data as
    good as possible.
  </p>
  <Plot maxWidth={800} domain={[-100, 100]} range={[-500, 500]}>
    <Ticks xTicks={[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]} yTicks={[-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]} xOffset={-15} yOffset={15}/>
    <XLabel text="Feature" fontSize={15}/>
    <YLabel text="Target" fontSize={15}/>
    <Circle data={linearData} radius={3} />
    <Path data={line} stroke={2}/>
  </Plot>

  <div class="flex-container">
    <div>
      <p><Latex>w</Latex>: {estimatedWeight}</p>
    </div>
    <Slider bind:value={estimatedWeight} min={-200} max={200} />
  </div>
  <div class="flex-container">
    <div>
      <p><Latex>b</Latex>: {estimatedBias}</p>
    </div>
    <Slider bind:value={estimatedBias} min={-500} max={500} />
  </div>
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
  <Latex>y = x_1w_1 + x_2w_2 + ... + x_nw_n+ b</Latex>
  <p>
    We can also use a more compact form and write the equation in vector form.
  </p>
  <Latex>{String.raw` y = \mathbf{x} \mathbf{w}^T + b \text{, where}`}</Latex>
  <Latex
    >{String.raw`
    \mathbf{x} = 
\begin{bmatrix}
   x_1 & x_2 & \cdots & x_n
\end{bmatrix}, 
\mathbf{w} = 
\begin{bmatrix}
   w_1 & 
   w_2 & 
   \cdots &
   w_n
\end{bmatrix}
`}
  </Latex>
  <p>
    In a three dimensional space we calculate a two dimensional plane that
    divides the coordinate system into two regions. This procedure is harder to
    imagine for more than 3 dimensions, but we still create a plane (a so called
    hyperplane) in space. The weights are used to rotate the hyperplane while
    the bias moves the plane.
  </p>
  <p>
    Usually we draw a "hat" over the <Latex>y</Latex> value to indicate that we are
    dealing with a prediction from a model,
    <Latex>{String.raw`\hat{y} = \mathbf{x} \mathbf{w}^T + b`}</Latex>. The <Latex
      >y</Latex
    > value on the other hand represents an actual target, the so called ground truth.
  </p>
  <p>
    In the next lectures we are going to cover how the learning procedure works.
    For now the main takeaway from this chapter should be the visual intuition
    of weights and biases.
  </p>
  <div class="separator" />
</Container>

<style>
  .flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .flex-container div {
    width: 100px;
  }
</style>
