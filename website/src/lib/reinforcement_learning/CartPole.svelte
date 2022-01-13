<script>
  //SOURCES FOR IMPLEMENTATION
  // https://github.com/tensorflow/tfjs-examples/blob/master/cart-pole/cart_pole.js
  // https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L117
  // https://coneural.org/florian/papers/05_cart_pole.pdf

  import { onMount } from "svelte";
  import Table from "$lib/Table.svelte";

  export let showState = true;

  let fps = 60;
  let fpsInterval = 1000 / fps;

  onMount(() => {
    reset();
    let then = Date.now();
    let frame;

    let loop = () => {
      frame = requestAnimationFrame(loop);
      let now = Date.now();
      let elapsed = now - then;
      if (elapsed >= fpsInterval) {
        then = now;
        let action = 0.1 * theta + 0.5 * thetaDot > 0 ? 1 : 0;
        takeAction(action);
      }
      return () => {
        cancelAnimationFrame(frame);
      };
    };

    requestAnimationFrame(loop);
  });
  // state
  let x = 0;
  let xDot = 0;
  let theta = 0;
  let thetaDot = 0;

  //reward
  let reward = 0;

  function reset() {
    x = Math.random() * (0.05 - -0.05) - 0.05;
    xDot = Math.random() * (0.05 - -0.05) - 0.05;
    theta = Math.random() * (0.05 - -0.05) - 0.05;
    thetaDot = Math.random() * (0.05 - -0.05) - 0.05;
    reward = 0;
  }

  //physics parameters from OpenAI gym
  const gravity = 9.8;
  const massCart = 1.0;
  const massPole = 0.1;
  const totalMass = massPole + massCart;
  const length = 0.5;
  const poleMassLength = massPole * length;
  const forceMag = 10.0;
  const tau = 0.02;

  const thetaThresholdRadians = (12 * 2 * Math.PI) / 360;
  const xThreshold = 2.4;

  //visual parameters
  export let width = 400;
  export let height = 100;

  const scale = width / (xThreshold * 2);

  const cartWidth = 0.3 * scale;
  const cartHeight = 0.12 * scale;
  $: cartX = x * scale + width / 2;
  const wheelRadius = cartHeight / 2.2;

  // this function is the JavaScript version of the OpenAI gym implementation
  function takeAction(action) {
    let force = action === 1 ? forceMag : -forceMag;
    let cosTheta = Math.cos(theta);
    let sinTheta = Math.sin(theta);

    let temp = (force + poleMassLength * thetaDot ** 2 * sinTheta) / totalMass;

    let thetaAcc =
      (gravity * sinTheta - cosTheta * temp) /
      (length * (4.0 / 3.0 - (massPole * cosTheta ** 2) / totalMass));

    let xAcc = temp - (poleMassLength * thetaAcc * cosTheta) / totalMass;

    x = x + tau * xDot;
    xDot = xDot + tau * xAcc;
    theta = theta + tau * thetaDot;
    thetaDot = thetaDot + tau * thetaAcc;

    if (checkDone()) {
      reset();
    } else {
      reward += 1;
    }
  }

  function checkDone() {
    return (
      x < -xThreshold ||
      x > xThreshold ||
      theta < -thetaThresholdRadians ||
      theta > thetaThresholdRadians
    );
  }

  // tabel data and header
  let header = ["Variable", "Value"];
  $: data = [
    ["Sum of Rewards", reward],
    ["Cart Position", x.toFixed(3)],
    ["Cart Velocity", xDot.toFixed(3)],
    ["Pole Angle", theta.toFixed(3)],
    ["Pole Angular Velocity", thetaDot.toFixed(3)],
  ];
</script>

<svg viewBox="0 0 {width} {height}">
  <!-- Cart -->
  <rect
    fill="none"
    stroke="var(--text-color)"
    x={cartX - cartWidth / 2}
    y={height - cartHeight - wheelRadius}
    width={cartWidth}
    height={cartHeight}
  />

  <!-- Wheels -->
  <g fill="var(--background-color)" stroke="var(--text-color)">
    <!-- Left -->
    <circle
      cx={cartX - cartWidth / 4}
      cy={height - wheelRadius}
      r={wheelRadius}
    />
    <!-- inside dot -->
    <circle cx={cartX - cartWidth / 4} cy={height - wheelRadius} r={0.5} />
    <!-- Right -->
    <circle
      cx={cartX + cartWidth / 4}
      cy={height - wheelRadius}
      r={wheelRadius}
    />
    <!-- inside dot -->
    <circle cx={cartX + cartWidth / 4} cy={height - wheelRadius} r={0.5} />
  </g>

  <!-- Pole -->
  <g
    transform="rotate({theta * (180 / Math.PI)} {cartX} {height -
      wheelRadius -
      cartHeight})"
  >
    <rect
      fill="none"
      stroke="var(--text-color)"
      x={cartX - 2}
      y={height - cartHeight - scale * length * 2 - wheelRadius}
      width={4}
      height={scale * length * 2}
    />
  </g>

  <!-- Ground -->
  <line x1={0} y1={height} x2={width} y2={height} stroke="var(--text-color)" />
</svg>

{#if showState}
  <Table {header} {data} />
{/if}
