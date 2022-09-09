<script>
  import Container from "$lib/Container.svelte";
  import Clipping from "../_gradient_clipping/Clipping.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";

  let valuePaths = [];
  function recalculateValue() {
    valuePaths = [];
    // original value
    let x = Math.random() * 6 - 3;
    let y = Math.random() * 6 - 3;

    valuePaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);

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
  }
  recalculateValue();

  let normPaths = [];
  function recalculateNorm() {
    normPaths = [];
    // original value
    let x = Math.random() * 6 - 3;
    let y = Math.random() * 6 - 3;

    normPaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);

    let norm = Math.sqrt(x ** 2 + y ** 2);
    if (norm > 1) {
      x = x / norm;
      y = y / norm;
    }

    normPaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);
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
    gradient descent. When you start the simulation, the gradients will start to
    move randomly in the 2d coordinate system. If one of the gradients is larger
    than one, we will clip that gradient to 1. So if one gradient is 3 and the
    other is 1.5, we clip both to 1, thereby disregarding the relative magnitude
    of the vector components and changing the direction of the vector. The
    clipped vector will move along the circumference of the square.
  </p>
  <PlayButton f={recalculateValue} delta={800} />
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-3, 3]}
    range={[-3, 3]}
  >
    <Ticks
      xTicks={[-3, -2, -1, 0, 1, 2, 3]}
      yTicks={[-3, -2, -1, 0, 1, 2, 3]}
    />
    <Path
      data={[
        { x: 1, y: 1 },
        { x: -1, y: 1 },
        { x: -1, y: -1 },
        { x: 1, y: -1 },
        { x: 1, y: 1 },
      ]}
      color="var(--main-color-1)"
    />
    <Path data={valuePaths[0]} strokeDashArray={[4, 4]} />
    <Path data={valuePaths[1]} color="var(--main-color-1)" />
  </Plot>
  <p>
    A better solution is to use norm clipping. When we clip the norm, we clip
    all the gradients proportialnally, such that the direction remains the same.
    Below we specifically use the L2 norm.
  </p>
  <Clipping type="norm" />
  <p>
    While the magnitude of the gradient vector is reduced to the threshold
    value, the direction remains unchanged. The clipped vector moves along the
    circumference of a circle.
  </p>
  <PlayButton f={recalculateNorm} delta={800} />
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-3, 3]}
    range={[-3, 3]}
  >
    <Ticks
      xTicks={[-3, -2, -1, 0, 1, 2, 3]}
      yTicks={[-3, -2, -1, 0, 1, 2, 3]}
    />
    <Circle data={[{ x: 0, y: 0 }]} color="none" radius="70" />
    <Path data={normPaths[0]} strokeDashArray={[4, 4]} />
    <Path data={normPaths[1]} color="var(--main-color-1)" />
  </Plot>
  <p>
    This solution feels like a hack, but it is quite pracktical. You might not
    be able to solve all your problems with gradient clipping, but it should be
    part of your toolbox.
  </p>
  <div class="separator" />
</Container>
