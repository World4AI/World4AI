<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Plot from "$lib/Plot.svelte";
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
  <div class="separator" />
  <h2>The Norm</h2>
  <p>
    We will start this section by reminding ourselves of a term usually learned
    in an introductory linear algebra course, the <Highlight>norm</Highlight>.
    The norm measures the distance of a vector
    <Latex>{String.raw`\mathbf{v}`}</Latex> from the origin. The p-norm (<Latex
      >L_p</Latex
    >) for the vector <Latex>{String.raw`\mathbf{v}`}</Latex> is defined as <Latex
      >{String.raw`||\mathbf{v}||_p = \large\sqrt{\sum_{i=1}^n |v_i|^p}`}</Latex
    >. By changing the parameter <Latex>p</Latex> from 1 to infinity we get different
    types of norms.
  </p>
  <p>
    In machine learning and deep learning we are especially interested in the<Latex
      >{String.raw`L_1`}</Latex
    > and the<Latex>{String.raw`L_2`}</Latex> norm.
  </p>
  <p>
    The <Latex>L_2</Latex> norm

    <Latex
      >{String.raw`
||\mathbf{v}||_2 = \sqrt{\sum_{i=1}^n v_i^2}
      `}</Latex
    >
    , also called the Euclidean norm is the distance measure we are most familiar
    with. Given the vector <Latex>{String.raw`\mathbf{v}`}</Latex> in<Latex
      >{String.raw`\mathbb{R}^2`}</Latex
    > for example the <Latex>L_2</Latex> norm <Latex
      >{String.raw`||\mathbf{v}||_2`}</Latex
    > is
    <Latex>{String.raw`\sqrt{v_1^2 + v_2^2}`}</Latex>. This is essentially the
    application of the Pythagorean theorem. When we are given the vector <Latex
      >{String.raw`\begin{bmatrix}
        5 \\
        4
      \end{bmatrix}`}</Latex
    > for example we can regard the number 5 as the x coordinate and the numver 4
    as the y coordinate. As shown in the graph below this vector creates a right
    triangle, therefore it makes sense to use the Pythagorean theorem to calculate
    the lengths of the vector.
  </p>
  <Plot
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

  <div class="separator" />
</Container>
