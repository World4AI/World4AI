<script>
  import { tweened } from "svelte/motion";
  import { draw } from "svelte/transition";
  import Question from "$lib/Question.svelte";
  import Button from "$lib/Button.svelte";

  let disabledNormal = false;
  const xTranslateNormal = tweened(0, {
    duration: 400,
  });

  const x1Normal = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const x2Normal = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const inputOpacityNormal = tweened(1, {
    duration: 400,
    delay: 500,
  });

  const processOpacityNormal = tweened(1, {
    duration: 400,
  });

  async function handleNormalProgramming() {
    disabledNormal = true;
    await xTranslateNormal.set(175);
    await processOpacityNormal.set(0);
    await processOpacityNormal.set(1);
    await x1Normal.set(80);
    await x2Normal.set(40);
    await xTranslateNormal.set(340);
    await inputOpacityNormal.set(0);
    x1Normal.set(60);
    x2Normal.set(60);
    await xTranslateNormal.set(0);
    await inputOpacityNormal.set(1);
    disabledNormal = false;
  }

  let disabledML = false;
  let showImprovement = false;
  let step = 0;
  let maxSteps = 3;
  let learned1 = false;
  let learned2 = false;
  let learned3 = false;

  const xTranslateML = tweened(0, {
    duration: 400,
  });

  const x1ML = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const x2ML = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const inputOpacityML = tweened(1, {
    duration: 400,
    delay: 500,
  });

  const processOpacityML = tweened(1, {
    duration: 400,
  });

  async function handleMLProgramming() {
    disabledML = true;
    step += 1;

    await xTranslateML.set(175);
    await processOpacityML.set(0);
    await processOpacityML.set(1);
    let y = ((80 - 60) * step) / maxSteps;
    await x1ML.set(60 + y);
    await x2ML.set(60 - y);
    await xTranslateML.set(340);
    showImprovement = true;
    if (step === 1) {
      learned1 = true;
    }
    if (step === 2) {
      learned2 = true;
    }
    if (step === 3) {
      learned3 = true;
    }
    await inputOpacityML.set(0);
    showImprovement = false;
    x1ML.set(60);
    x2ML.set(60);
    await xTranslateML.set(0);
    await inputOpacityML.set(1);

    if (step === 3) {
      step = 0;
      learned1 = false;
      learned2 = false;
      learned3 = false;
    }
    disabledML = false;
  }
</script>

<h1>Machine Learning</h1>
<Question>What is machine learning?</Question>
<div class="separator" />
<p>
  The term <strong>Machine Learning</strong> is usually attributed to Arthur Samuel,
  who defined it as follows.
</p>
<p class="info">
  "Machine learining is the field of study that gives computers the ability to
  learn without being explicitly programmed<sup>[1]</sup>".
</p>

<p>
  We will use a slightly more programming oriented definition of machine
  learning.
</p>
<p class="info">
  Machine learning is a programming paradigm, where the logic of a program is
  learned from data.
</p>

<svg version="1.1" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
  <g stroke="var(--text-color)">
    <g id="logic">
      <g opacity={$processOpacityNormal} fill="#fff">
        <rect x="153.25" y="105" width="8.4993" height="8.5008" />
        <rect x="204.25" y="105" width="8.4993" height="8.5008" />
        <rect x="238.24" y="105" width="8.4993" height="8.5008" />
        <rect x="187.25" y="122" width="8.4993" height="8.5008" />
        <rect x="170.25" y="139" width="8.4993" height="8.5008" />
        <rect x="238.24" y="139" width="8.4993" height="8.5008" />
        <rect x="153.25" y="156" width="8.4993" height="8.5008" />
        <rect x="204.25" y="156" width="8.4993" height="8.5008" />
        <rect x="187.25" y="173.01" width="8.4993" height="8.5008" />
        <rect x="204.25" y="173.01" width="8.4993" height="8.5008" />
      </g>
      <g fill="none">
        <rect x="149" y="100.75" width="101.99" height="85.008" />
        <rect x="170.25" y="105" width="8.4993" height="8.5008" />
        <rect x="187.25" y="105" width="8.4993" height="8.5008" />
        <rect x="221.24" y="105" width="8.4993" height="8.5008" />
        <rect x="153.25" y="122" width="8.4993" height="8.5008" />
        <rect x="170.25" y="122" width="8.4993" height="8.5008" />
        <rect x="204.25" y="122" width="8.4993" height="8.5008" />
        <rect x="221.24" y="122" width="8.4993" height="8.5008" />
        <rect x="238.24" y="122" width="8.4993" height="8.5008" />
        <rect x="153.25" y="139" width="8.4993" height="8.5008" />
        <rect x="187.25" y="139" width="8.4993" height="8.5008" />
        <rect x="204.25" y="139" width="8.4993" height="8.5008" />
        <rect x="221.24" y="139" width="8.4993" height="8.5008" />
        <rect x="221.24" y="156" width="8.4993" height="8.5008" />
        <rect x="238.24" y="156" width="8.4993" height="8.5008" />
        <rect x="153.25" y="173.01" width="8.4993" height="8.5008" />
        <rect x="170.25" y="173.01" width="8.4993" height="8.5008" />
        <rect x="170.25" y="156" width="8.4993" height="8.5008" />
        <rect x="187.25" y="156" width="8.4993" height="8.5008" />
        <rect x="221.24" y="173.01" width="8.4993" height="8.5008" />
        <rect x="238.24" y="173.01" width="8.4993" height="8.5008" />
      </g>
    </g>
    <rect
      id="transformer"
      x="150"
      y="19.992"
      width="99.992"
      height="75.007"
      fill="none"
    />
    <g
      opacity={$inputOpacityNormal}
      transform="translate({$xTranslateNormal}, 0)"
    >
      <polygon fill="none" points="1,{$x1Normal} 1,{$x2Normal} 51,40 51,80" />
    </g>
  </g>
</svg>
<Button
  on:click={handleNormalProgramming}
  disabled={disabledNormal}
  value="RUN"
/>

<svg version="1.1" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
  <g stroke="var(--text-color)">
    <g id="logic">
      <g opacity={$processOpacityML} fill="#fff">
        <rect
          x="153.25"
          y="105"
          width="8.4993"
          height="8.5008"
          fill={learned1 === true ? "var(--main-color-2)" : "none"}
        />
        <rect
          x="204.25"
          y="105"
          width="8.4993"
          height="8.5008"
          fill={learned2 === true ? "var(--main-color-2)" : "none"}
        />
        <rect x="238.24" y="105" width="8.4993" height="8.5008" />
        <rect x="187.25" y="122" width="8.4993" height="8.5008" />
        <rect x="170.25" y="139" width="8.4993" height="8.5008" />
        <rect x="238.24" y="139" width="8.4993" height="8.5008" />
        <rect
          x="153.25"
          y="156"
          width="8.4993"
          height="8.5008"
          fill={learned3 === true ? "var(--main-color-2)" : "none"}
        />
        <rect x="204.25" y="156" width="8.4993" height="8.5008" />
        <rect x="187.25" y="173.01" width="8.4993" height="8.5008" />
        <rect x="204.25" y="173.01" width="8.4993" height="8.5008" />
      </g>
      <g fill="none">
        <rect x="149" y="100.75" width="101.99" height="85.008" />
        <rect x="170.25" y="105" width="8.4993" height="8.5008" />
        <rect x="187.25" y="105" width="8.4993" height="8.5008" />
        <rect x="221.24" y="105" width="8.4993" height="8.5008" />
        <rect x="153.25" y="122" width="8.4993" height="8.5008" />
        <rect x="170.25" y="122" width="8.4993" height="8.5008" />
        <rect x="204.25" y="122" width="8.4993" height="8.5008" />
        <rect x="221.24" y="122" width="8.4993" height="8.5008" />
        <rect x="238.24" y="122" width="8.4993" height="8.5008" />
        <rect x="153.25" y="139" width="8.4993" height="8.5008" />
        <rect x="187.25" y="139" width="8.4993" height="8.5008" />
        <rect x="204.25" y="139" width="8.4993" height="8.5008" />
        <rect x="221.24" y="139" width="8.4993" height="8.5008" />
        <rect x="221.24" y="156" width="8.4993" height="8.5008" />
        <rect x="238.24" y="156" width="8.4993" height="8.5008" />
        <rect x="153.25" y="173.01" width="8.4993" height="8.5008" />
        <rect x="170.25" y="173.01" width="8.4993" height="8.5008" />
        <rect x="170.25" y="156" width="8.4993" height="8.5008" />
        <rect x="187.25" y="156" width="8.4993" height="8.5008" />
        <rect x="221.24" y="173.01" width="8.4993" height="8.5008" />
        <rect x="238.24" y="173.01" width="8.4993" height="8.5008" />
      </g>
    </g>
    <rect
      id="transformer"
      x="150"
      y="19.992"
      width="99.992"
      height="75.007"
      fill="none"
    />

    <g
      opacity="1"
      stroke="var(--main-color-1)"
      stroke-dasharray="4 2"
      transform="translate(340, 0)"
    >
      <polygon fill="none" points="1,80 1,40 51,40 51,80" />
    </g>

    <g opacity={$inputOpacityML} transform="translate({$xTranslateML}, 0)">
      <polygon fill="none" points="1,{$x1ML} 1,{$x2ML} 51,40 51,80" />
    </g>

    {#if showImprovement}
      <path
        in:draw={{ duration: 1000 }}
        d="M 365 90 V 150 H 260"
        stroke="red"
        fill="none"
      />
    {/if}
  </g>
</svg>
<Button on:click={handleMLProgramming} disabled={disabledML} value="RUN" />
<div class="separator" />

<h2>Notes</h2>
<div class="notes">
  <p>
    [1] This definition is supposedly based on Samuel, A.L (1959), Some studies
    in machine learning using the game of checkers, IBM Journal of research and
    development, 2(2), 210-229. The exact quote is not contained in that paper,
    only the general sentiment regarding that definition.
  </p>
</div>
<div class="separator" />

<style>
  .notes p {
    line-height: 1;
    font-size: 15px;
    opacity: 80%;
  }
</style>
