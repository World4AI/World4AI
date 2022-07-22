<script>
  import Container from "$lib/Container.svelte";
  import Clipping from "./_gradient_clipping/Clipping.svelte";
  import Plot from "$lib/Plot.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  let valueIntervalId = null;
  let valuePaths = [];
  let valuePoints = [];
  function recalculateValue() {
    valuePaths = [];
    valuePoints = [];
    // original value
    let x = Math.random() * 6 - 3;
    let y = Math.random() * 6 - 3;

    valuePaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);
    valuePoints.push([{ x, y }]);

    //clip values
    if (x >= 1) {
      x = 1;
    } else if (x < -1) {
      x = -1;
    }
    if (y >= 1) {
      y = 1;
    } else if (y < -1) {
      y = -1;
    }

    valuePaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);
    valuePoints.push([{ x, y }]);
  }
  recalculateValue();

  let normIntervalId = null;
  let normPaths = [];
  let normPoints = [];
  function recalculateNorm() {
    normPaths = [];
    normPoints = [];
    // original value
    let x = Math.random() * 6 - 3;
    let y = Math.random() * 6 - 3;

    normPaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);
    normPoints.push([{ x, y }]);

    let norm = Math.sqrt(x ** 2 + y ** 2);
    x = x / norm;
    y = y / norm;

    normPaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);
    normPoints.push([{ x, y }]);
  }
  recalculateNorm();
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Gradient Clipping</title>
  <meta
    name="description"
    content="Gradient clipping clips either individual gradient values or the norm of the gradient vector at a predetermined threshold, thereby reducing the likelihood of exploding gradients."
  />
</svelte:head>

<h1>Gradient Clipping</h1>
<div class="separator" />

<Container>
  <p>
    Exploding gradients is a problem where the backpropagation algorithm returns
    larger and larger gradients. The algorithm gets increasingly more unstable
    until the gradient values do not fit into memory.
  </p>
  <p>
    So why do we not just determine a gradient threshold and if the gradient
    value moves beyond that threshold we cut the gradient? The technique we just
    described is called gradient clipping, value clipping to be exact. At the
    start of the training process we determine a value beyond which the absolute
    value of the gradient is not allowed to move.
  </p>
  <p>
    The interactive example below demonstrates the process. You can use the
    slider to move the threshold for the gradient clipping. When individual
    gradients move outside the range, we clip them.
  </p>
  <Clipping type="value" />
  <p>
    Value clipping is problematic, because it basically changes the direction of
    gradient descent. The play button will start to move the gradients around
    the 2d coordinate system. If ones of the vectors is shorter, that means that
    the vector was clipped. In most cases that means that the original (longer)
    vector and the clipped vectors will show into different directions.
  </p>
  <PlayButton f={recalculateValue} delta={800} />
  <Plot
    pathsData={valuePaths}
    pointsData={valuePoints}
    config={{
      width: 500,
      height: 500,
      maxWidth: 500,
      minX: -3,
      maxX: 3,
      minY: -3,
      maxY: 3,
      xLabel: "Weight 1",
      yLabel: "Weight 2",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 2,
      numTicks: 7,
    }}
  />
  <p>
    The solution is to use norm clipping. When we clip the norm, we clip all the
    gradients proportialnally, such that the direction remains the same. Below
    we specifically use the L2 norm.
  </p>
  <Clipping type="norm" />
  <p>
    While the magnitude of the gradient vector is reduced to the threshold
    value, the direction remains unchanged.
  </p>
  <PlayButton f={recalculateNorm} delta={800} />
  <Plot
    pathsData={normPaths}
    pointsData={normPoints}
    config={{
      width: 500,
      height: 500,
      maxWidth: 500,
      minX: -3,
      maxX: 3,
      minY: -3,
      maxY: 3,
      xLabel: "Weight 1",
      yLabel: "Weight 2",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 2,
      numTicks: 7,
    }}
  />
  <p>
    This solution feels like a hack, but it is quite pracktical. You might not
    be able to solve all your problems with gradient clipping, but it should be
    part of your toolbox.
  </p>
</Container>
