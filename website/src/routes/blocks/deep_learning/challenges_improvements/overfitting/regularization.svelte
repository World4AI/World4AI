<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Plot from "$lib/Plot.svelte";
  import PlayButton from "$lib/PlayButton.svelte";
  import Slider from "$lib/Slider.svelte";
  import L1Polynomial from "./_regularization/L1Polynomial.svelte";

  // circles of different size
  function generateUnit(lengths = [1]) {
    let paths = [];
    lengths.forEach((len) => {
      let path = [];
      for (let i = 0; i <= Math.PI * 2; i += 0.01) {
        let x = Math.cos(i) * len;
        let y = Math.sin(i) * len;
        path.push({ x, y });
      }
      paths.push(path);
    });
    return paths;
  }

  let unitPath = generateUnit([1]);
  let unitPlusVectorPath;

  let unitPoint;
  let circularAngle = 0.5;
  function moveCircular() {
    circularAngle += 0.1;
    let x = Math.cos(circularAngle);
    let y = Math.sin(circularAngle);
    unitPoint = [{ x, y }];
    unitPlusVectorPath = [[{ x: 0, y: 0 }, ...unitPoint], ...unitPath];
  }
  moveCircular();

  let moveCircleIntervalId = null;
  function rotationHandler() {
    if (!moveCircleIntervalId) {
      moveCircleIntervalId = setInterval(moveCircular, 100);
    } else {
      clearInterval(moveCircleIntervalId);
      moveCircleIntervalId = null;
    }
  }
  let manyCircles = generateUnit([1, 2, 3]);

  // draw equation of the form x_1 + x_2 = 3
  let infiniteSolutions = [];

  for (let i = -4; i <= 4; i++) {
    let x = i;
    let y = 3 - i;
    infiniteSolutions.push({ x, y });
  }

  let radius = 0.1;
  $: touchingCircle = [infiniteSolutions, ...generateUnit([radius])];
</script>

<h1>Regularization</h1>
<div class="separator" />
<Container>
  <p>
    The goal of regularization is to encourage a simpler model. Simpler models
    can not fit to the exact form of the data and are therefore less prone to
    overfitting. While there are several techniques to achieve that goal, in
    this section we focus on techniques that modify the loss function.
  </p>
  <p>
    We will start this section by reminding ourselves of a term usually learned
    in an introductory linear algebra course, the <Highlight>norm</Highlight>.
    The norm measures the distance of a vector
    <Latex>{String.raw`\mathbf{v}`}</Latex> from the origin. The p-norm (written
    <Latex>L_p</Latex>) for the vector <Latex>{String.raw`\mathbf{v}`}</Latex> is
    defined as <Latex
      >{String.raw`||\mathbf{v}||_p = \large\sqrt{\sum_{i=1}^n |v_i|^p}`}</Latex
    >. By changing the parameter <Latex>p</Latex> from 1 to infinity we get different
    types of norms. In machine learning and deep learning we are especially interested
    in the so called <Latex>{String.raw`L_1`}</Latex> and the<Latex
      >{String.raw`L_2`}</Latex
    > norm.
  </p>
  <p>
    The <Latex>L_2</Latex> norm also called Euclidean norm is the distance measure
    we are most familiar with. When we are given the vector <Latex
      >{String.raw`\begin{bmatrix}
        5 \\
        4
      \end{bmatrix}`}</Latex
    > for example we can regard the number 5 as the x coordinate and the numver 4
    as the y coordinate. As shown in the graph below this vector is essentially a
    hypotenuse of a right triangle, therefore we need to apply the Pythagorean theorem
    to calculate the length of the vector, <Latex
      >{String.raw`c^2 = \sqrt{a^2 + b^2}`}</Latex
    >.
  </p>

  <Plot
    pointsData={[{ x: 5, y: 4 }]}
    pathsData={[
      [
        { x: 0, y: 0 },
        { x: 5, y: 0 },
      ],
      [
        { x: 5, y: 0 },
        { x: 5, y: 4 },
      ],
      [
        { x: 0, y: 0 },
        { x: 5, y: 4 },
      ],
    ]}
    config={{
      width: 500,
      height: 500,
      maxWidth: 450,
      minX: 0,
      maxX: 5,
      minY: 0,
      maxY: 5,
      xLabel: "x",
      yLabel: "y",
      xTicks: [1, 2, 3, 4, 5],
      yTicks: [1, 2, 3, 4, 5],
    }}
  />

  <p>
    While the the Pythagorean theorem is used to calculate the length of a
    vector on a 2 dimensional plane, the <Latex>L_2</Latex> norm generalizes to
    <Latex>n</Latex> dimensions,
    <Latex
      >{String.raw`
||\mathbf{v}||_2 = \sqrt{\sum_{i=1}^n v_i^2}
      `}</Latex
    >, giving us the ability to calculate the distance of the vector from the
    origin in arbitrary dimension.
  </p>
  <p />
  <p>
    If we are given a specific <Latex>L_2</Latex> norm for a vector <Latex
      >l</Latex
    > such that <Latex>{String.raw`\sqrt{x_1^2 + x_2^2} = l`}</Latex>, we will
    find that there is whole set of vectors that that satisfy that condition and
    that this set has a circular shape. In the interactive example below we draw
    a unit circle, such that <Latex
      >{String.raw`\sqrt{x_1^2 + x_2^2} = 1`}</Latex
    >. When the <Latex>x_1</Latex> variable changes, the <Latex>x_2</Latex> variable
    has to move along in order for the sum of the squares to amount to 1.
  </p>
  <PlayButton
    type={!moveCircleIntervalId ? "play" : "pause"}
    on:click={rotationHandler}
  />
  <Plot
    pathsData={unitPlusVectorPath}
    pointsData={unitPoint}
    config={{
      width: 500,
      height: 500,
      maxWidth: 500,
      minX: -1.5,
      maxX: 1.5,
      minY: -1.5,
      maxY: 1.5,
      xLabel: "x 1",
      yLabel: "x 2",
      xTicks: [-1, 0, 1],
      yTicks: [-1, 0, 1],
    }}
  />
  <p>
    There is an infinite number of such circles, each corresponding to a
    different set of vectors with a particular <Latex>L_2</Latex> norm. The larger
    the radius, the larger the norm. Below we draw the circles that correspond to
    the <Latex>L_2</Latex> norm of 1, 2 and 3 respectively.
  </p>
  <Plot
    pathsData={manyCircles}
    config={{
      width: 500,
      height: 500,
      maxWidth: 500,
      minX: -4,
      maxX: 4,
      minY: -4,
      maxY: 4,
      xLabel: "x 1",
      yLabel: "x 2",
      xTicks: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
      yTicks: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    }}
  />
  <p>
    Now lets try and anderstand how this visualization of a norm can be useful.
    Imagine we are given a single equation <Latex
      >{String.raw`x_1 + x_2 = 3`}</Latex
    >. This is an underdetermined system of equations, because we have just 1
    equation and 2 unknowns. That means that there is literally an infinite
    number of possible solutions. We can depict the whole set of solutions as a
    single line. Points on that line show combinations of <Latex>x_1</Latex> and
    <Latex>x_2</Latex> that sum up to 3.
  </p>
  <Plot
    pathsData={infiniteSolutions}
    config={{
      width: 500,
      height: 500,
      maxWidth: 500,
      minX: -4,
      maxX: 4,
      minY: -4,
      maxY: 4,
      xLabel: "x 1",
      yLabel: "x 2",
      xTicks: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
      yTicks: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    }}
  />
  <p>
    Below we add the norm for the <Latex>x_1</Latex>, <Latex>x_2</Latex> vector.
    The slider controls the size of the <Latex>L_2</Latex> norm. When you keep increasing
    the norm you will realize that at the red point the circle is going to just barely
    touch the line. At this point we see the solution for <Latex
      >{String.raw`x_1 + x_2 = 3`}</Latex
    > that has the smallest <Latex>L2</Latex> norm out of all possible solutions.
  </p>
  <Plot
    pathsData={touchingCircle}
    pointsData={[{ x: 1.5, y: 1.5 }]}
    config={{
      width: 500,
      height: 500,
      maxWidth: 500,
      minX: -4,
      maxX: 4,
      minY: -4,
      maxY: 4,
      xLabel: "x 1",
      yLabel: "x 2",
      xTicks: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
      yTicks: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    }}
  />
  <Slider bind:value={radius} min="0.1" max="3" step="0.1" />
  <p>
    If we are able to find solutions that have a comparatively low <Latex
      >L_2</Latex
    > how does this apply to machine learning and why is this useful to avoid overfitting?
    We can add the squared <Latex>L_2</Latex> norm to the loss function as a regularizer.
    We square the norm to avoid the root and to make the calculation of the derivative
    easier.
  </p>
  <p>
    If we are dealing with the mean squared error for example our new function
    looks as below.
  </p>
  <Latex
    >{String.raw`L=\frac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)} )^2 + \lambda \sum_j^m w_j^2`}</Latex
  >
  <p>
    The overall idea is to find the solution that reduces the mean squared error
    without creating large weights. When the size of one of the weights increses
    disproportionatly, the loss function will rise sharply. Therefore by using
    the regularization term we reduce the overemphasis on any particular
    feature, thereby reducing the complexity of the model. The <Latex
      >\lambda</Latex
    > (lambda) is the hyperparameter that we can tune to determine how much emphasis
    we would like to put on the <Latex>L_2</Latex> norm. It is the lever that lets
    you control the size of the weights.
  </p>
  <p>
    Below you can move the slider to adjust the lambda. The higher the lambda,
    the simpler the model becomes and the more the curve looks like a straight
    line.
  </p>
  <L1Polynomial />
  <div class="separator" />
</Container>
