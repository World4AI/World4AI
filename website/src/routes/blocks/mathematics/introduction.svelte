<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Mathematics | Introduction</title>
  <meta
    name="description"
    content="Artificial Intelligence requires fundamental mathematical skills, like linear algebra, calculus and probability theory."
  />
</svelte:head>

<h1>Introduction</h1>
<Question>Why is mathematics important for artificial intelligence?</Question>
<div class="separator" />
<svg version="1.1" viewBox="0 0 400 350" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="TriangleOutL" overflow="visible" orient="auto">
      <path
        transform="scale(.8)"
        d="m5.77 0-8.65 5v-10z"
        fill="var(--text-color)"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g stroke="var(--text-color)">
    <path
      d="m25 345v-150"
      fill="none"
      marker-end="url(#TriangleOutL)"
      stroke-width="1px"
    />
    <path
      d="m5 325h150"
      fill="none"
      marker-end="url(#TriangleOutL)"
      stroke-width="1px"
    />
    <g transform="translate(5,-135)" fill="none" stroke-width="1px">
      <path d="m40 340s0 100 40 100" />
      <path d="m120 340s0 100-40 100" />
    </g>
    <path
      d="m102 320 38-130"
      fill="none"
      stroke-dasharray="1.5, 1.5"
      stroke-width=".5"
    />
    <circle
      cx="117.5"
      cy="267.5"
      r="1.8216"
      fill="#fff"
      stroke-width=".36432"
    />
  </g>
  <g fill="none" stroke="var(--text-color)">
    <g stroke-width="1px">
      <path d="m25 160v-150" marker-end="url(#TriangleOutL)" />
      <path d="m5 140h150" marker-end="url(#TriangleOutL)" />
      <path
        d="m25 50c13.334-3.3335 26.667-6.6668 41.668-3.333 15 3.3339 31.667 13.334 51.667 13.333 20-5.56e-4 43.333-10 66.666-20"
      />
    </g>
    <g stroke-width=".5">
      <path d="m50 45v95" />
      <path d="m140 57v83" />
      <path d="m55 45v95" />
      <path d="m60 45v95" />
      <path d="m65 46v94" />
      <path d="m70 47v93" />
      <path d="m75 49v91" />
      <path d="m80 51v89" />
      <path d="m85 53v87" />
      <path d="m90 54v86" />
      <path d="m95 56v84" />
      <path d="m100 57v83" />
      <path d="m105 59v81" />
      <path d="m110 59v81" />
      <path d="m115 60v80" />
      <path d="m120 60v80" />
      <path d="m125 60v80" />
      <path d="m130 59v81" />
      <path d="m135 58v82" />
    </g>
  </g>
  <g fill="none" stroke="var(--text-color)">
    <path
      d="m260 345v-150"
      marker-end="url(#TriangleOutL)"
      stroke-width="1px"
    />
    <path d="m240 325h150" marker-end="url(#TriangleOutL)" stroke-width="1px" />
    <path
      d="m260 325 25-85"
      marker-end="url(#TriangleOutL)"
      stroke-width=".5"
    />
    <path
      d="m260 325 110-115"
      marker-end="url(#TriangleOutL)"
      stroke-width=".5"
    />
    <g stroke-dasharray="2, 1" stroke-width=".5">
      <path d="m260 240h25" />
      <path d="m285 240v85" />
      <path d="m260 210h110" />
      <path d="m370 210v115" />
    </g>
  </g>
  <g fill="none" stroke="var(--text-color)" stroke-width="1px">
    <path d="m260 160v-150" marker-end="url(#TriangleOutL)" />
    <path d="m240 140h150" marker-end="url(#TriangleOutL)" />
  </g>
  <path
    d="m348.39 51.612-88.388 88.388h-40"
    fill="none"
    stroke="var(--main-color-1)"
    stroke-width="1px"
  />
</svg>
<p>
  The question whether we need to learn mathematics in order to understand
  artificial intelligence seems strange, because for the most part <Highlight
    >artificial intelligence is mathematics</Highlight
  >. If we take deep learning for example, the training of a neural network
  consists of matrix multiptlications, non linear activation functions, gradient
  descent and so on. Admittedly, at this point in time we can get away without
  an in-depth knowledge of mathematics by utilizing one of the deep learning
  frameworks and we could still produce good results. If you are an absolute
  beginner, then this approach is not necessarily bad. You can gather some
  practical experience and feel a sense of accoplishment. But the farther you
  progress, the more you will realize that a good grasp of mathematical methods
  is a prerequisite to become proficient in the field of artificial
  intelligence. The good part is that you can learn mathematics and machine
  learning in parallel. That way you have a way to relate abstract mathematical
  ideas to practical ai applications. For example if you study deep learning,
  linear algebra makes a lot more sense. If on the other hand you study
  reinforcement learning, Markov chains will become more relevant. You can
  choose for yourself if you want to dive deep into the mathematics or if you
  would prefer to take a look at deep learning and return for the relevant
  topics once the need arises.
</p>
<div class="separator" />
