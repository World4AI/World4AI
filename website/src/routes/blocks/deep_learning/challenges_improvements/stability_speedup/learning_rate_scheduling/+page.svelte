<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Title from "$lib/plt/Title.svelte";

  import PlayButton from "$lib/button/PlayButton.svelte";

  let data = [];
  for (let i = -8; i <= 8; i += 0.1) {
    data.push({ x: i, y: i ** 2 });
  }

  let xCoordinateFixedLR = 8;
  $: yCoordinateFixedLR = xCoordinateFixedLR ** 2;
  let alphaFixed = 0.1;
  let momentumFixed = 0;
  let prevYFixed = yCoordinateFixedLR;

  function gradientDescentStepFixed() {
    prevYFixed = yCoordinateFixedLR;
    let beta = 0.95;
    let grad = 2 * xCoordinateFixedLR;
    if (momentumFixed === 0) {
      momentumFixed = grad;
    }
    momentumFixed = momentumFixed * beta + grad * (1 - beta);
    xCoordinateFixedLR -= alphaFixed * momentumFixed;
  }

  let xCoordinateMovingLR = 8;
  $: yCoordinateMovingLR = xCoordinateMovingLR ** 2;
  let alphaMoving = 0.1;
  let momentumMoving = 0;
  let prevYMoving = yCoordinateMovingLR;

  function gradientDescentStepMoving() {
    if (xCoordinateMovingLR ** 2 > prevYMoving) {
      alphaMoving *= 0.88;
    }
    prevYMoving = yCoordinateMovingLR;
    let beta = 0.95;
    let grad = 2 * xCoordinateMovingLR;
    if (momentumMoving === 0) {
      momentumMoving = grad;
    }
    momentumMoving = momentumMoving * beta + grad * (1 - beta);
    xCoordinateMovingLR -= alphaMoving * momentumMoving;
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Learning Rate Scheduling</title>
  <meta
    name="description"
    content="Learning rate schedulers, also called learning rate decay, are techniques that change the learning rate over time."
  />
</svelte:head>

<h1>Learning Rate Scheduling</h1>
<div class="separator" />

<Container>
  <p>
    There is probably no hyperparameter that is more important than the learning
    rate <Latex>\alpha</Latex>. If the learning rate is too high, we might
    overshood or oscilate. If the learning rate is too low, training might be
    too slow, or we might get stuck in some local minimum.
  </p>
  <p>
    In the example below we pick a learning rate that is relatively large and we
    use gradient descent with momentum. The gradient descent algorithm
    overshoots and keeps oscilating for a while, before settling on the minimum.
  </p>
  <PlayButton f={gradientDescentStepFixed} delta={50} />
  <Plot domain={[-8, 8]} range={[0, 60]}>
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 10, 20, 30, 40, 50, 60]}
    />
    <Path {data} />
    <Circle data={[{ x: xCoordinateFixedLR, y: yCoordinateFixedLR }]} />
    <Title text={`Constant Learning Rate ${alphaFixed.toFixed(2)}`} />
  </Plot>

  <p>
    It is possible, that a single constant rate is not the optimal solution.
    What if we start out with a relatively large learning rate to gain momentum
    at the beginning of trainig, but decrease the learning rate either over time
    or at specific events. In deep learning this is called <Highlight
      >learning rate decay</Highlight
    > or <Highlight>learning rate scheduling</Highlight>. There are dozens of
    different schedulers. You could for example decay the learing rate by
    subtracting a constant rate every <Latex>n</Latex> episodes. Or you could multiply
    the learning rate at the end of each epoch by a constant factor, for example
    <Latex>0.9</Latex>. Below we use a popular learning rate decay technique
    that is called <Highlight>reduce learning rate on plateau</Highlight>. Once
    a metric (like a loss) stops improving for certain amount of epochs we
    decrease the learning rate by a predetermined factor. Below we reduce the
    learning rate on plateau, which makes the ball "glide" into the optimal
    value.
  </p>
  <PlayButton f={gradientDescentStepMoving} delta={50} />
  <Plot domain={[-8, 8]} range={[0, 60]}>
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 10, 20, 30, 40, 50, 60]}
    />
    <Path {data} />
    <Circle data={[{ x: xCoordinateMovingLR, y: yCoordinateMovingLR }]} />
    <Title text={`Variable Learning Rate ${alphaMoving.toFixed(3)}`} />
  </Plot>
  <p>
    Deep learning frameworks like PyTorch or Keras make it extremely easy to
    create learning rate schedulers. Usually it involves no more than 2-3 lines
    of code.
  </p>
  <p>
    There are no hard regarding learning rate decay. You can experiment and see
    what works for you.
  </p>
  <div class="separator" />
</Container>
