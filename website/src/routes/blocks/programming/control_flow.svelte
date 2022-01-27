<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";

  let height = 450;
  let width = 200;
  let radius = 10;
  let topOffset = 1;

  let simpleFlow = [
    { x: width / 2, y: radius + topOffset },
    { x: width / 2, y: radius + topOffset + 100 },
    { x: width / 2, y: radius + topOffset + 200 },
    { x: width / 2, y: radius + topOffset + 300 },
    { x: width / 2, y: radius + topOffset + 400 },
  ];

  let simpleConnections = [];
  simpleFlow.forEach((address, idx) => {
    if (idx != 0) {
      let coordinates = {};
      coordinates.x2 = address.x;
      coordinates.y2 = address.y;
      coordinates.x1 = simpleFlow[idx - 1].x;
      coordinates.y1 = simpleFlow[idx - 1].y;
      simpleConnections.push(coordinates);
    }
  });

  let controlledFlow = [
    { x: width / 2, y: radius + topOffset },
    { x: width / 2 - 50, y: radius + topOffset + 100, connection: 0 },
    { x: width / 2 + 50, y: radius + topOffset + 100, connection: 0 },
    { x: width / 2 - 50, y: radius + topOffset + 200, connection: 1 },
    { x: width / 2 + 50, y: radius + topOffset + 200, connection: 2 },
    {
      x: width / 2 - 50,
      y: radius + topOffset + 300,
      connection: 3,
      backConnection: 3,
    },
    { x: width / 2 + 50, y: radius + topOffset + 300, connection: 4 },
    { x: width / 2 - 50, y: radius + topOffset + 400, connection: 5 },
    {
      x: width / 2 + 50,
      y: radius + topOffset + 400,
      connection: 6,
      backConnection: 4,
    },
  ];

  let controlledConnections = [];
  controlledFlow.forEach((address) => {
    if (
      typeof address.connection != "undefined" &&
      typeof address.backConnection != "undefined"
    ) {
      let coordinates = {};
      coordinates.x2 = address.x - 3;
      coordinates.y2 = address.y;
      coordinates.x1 = controlledFlow[address.connection].x - 3;
      coordinates.y1 = controlledFlow[address.connection].y;
      controlledConnections.push(coordinates);
    }
    if (typeof address.backConnection != "undefined") {
      let coordinates = {};
      coordinates.x2 = address.x + 3;
      coordinates.y2 = address.y;
      coordinates.x1 = controlledFlow[address.connection].x + 3;
      coordinates.y1 = controlledFlow[address.connection].y;
      controlledConnections.push(coordinates);
    }
    if (
      typeof address.connection != "undefined" &&
      typeof address.backConnection == "undefined"
    ) {
      let coordinates = {};
      coordinates.x2 = address.x;
      coordinates.y2 = address.y;
      coordinates.x1 = controlledFlow[address.connection].x;
      coordinates.y1 = controlledFlow[address.connection].y;
      controlledConnections.push(coordinates);
    }
  });
</script>

<svelte:head>
  <title>World4AI | Programming | Control Flow</title>
  <meta
    name="description"
    content="Control flow allows programmers to produce programs with branching logic and code that can be repeated many hundrets of times. Conditions and loops are the two main tools of control flow."
  />
</svelte:head>

<h1>Control Flow</h1>
<Question
  >What do we mean when we talk about control flow in programming?</Question
>
<div class="separator" />
<p>
  Before we dive into the theory and practice of control flow, let us try to
  intuitively understand what we are missing in our programming so far and how
  control flow might be helpful.
</p>
<p>
  On the left side we see the the types of programs that we learned to create so
  far. The program is executed from top to bottom without any deviations.
</p>
<p>
  The right side on the other hand depicts a typical program that needs to
  implement some type of logic. The execution is not done from top to bottom in
  a strictly linear manner, but there is branching and processes to go a step
  back and repeat a certain process. This is what we mean when we talk about
  control flow: Python as many other programming languages provides
  functionality to control the execution flow of a program.
</p>
<div class="flex-space">
  <svg viewBox="0 0 {width} {height}">
    {#each simpleConnections as coordinates}
      <line
        x1={coordinates.x1}
        y1={coordinates.y1}
        x2={coordinates.x2}
        y2={coordinates.y2}
        stroke="var(--text-color)"
      />
    {/each}
    {#each simpleFlow as address}
      <circle
        fill="var(--main-color-1)"
        stroke="var(--text-color)"
        stroke-width="2px"
        cx={address.x}
        cy={address.y}
        r={radius}
      />
    {/each}
  </svg>
  <svg viewBox="0 0 {width} {height}">
    {#each controlledConnections as coordinates}
      <line
        x1={coordinates.x1}
        y1={coordinates.y1}
        x2={coordinates.x2}
        y2={coordinates.y2}
        stroke="var(--text-color)"
      />
    {/each}
    {#each controlledFlow as address}
      <circle
        fill="var(--main-color-2)"
        stroke="var(--text-color)"
        stroke-width="2px"
        cx={address.x}
        cy={address.y}
        r={radius}
      />
    {/each}
  </svg>
</div>
<p>
  Without control flow there would be no way of creating programs of any
  complexity. For example it is a common practice to execute a certain part of
  the code, only when a specific condition is met. Or we need to call a function
  1,000,000 times with different parameters. Both tasks can be easily
  accomplished with control flow techniques.
</p>
<p>
  When we talk about control flow, we generally talk about <Highlight
    >conditions</Highlight
  >: the ability to create branching and <Highlight>loops</Highlight>: the
  ability to repeat a certain piece of code several times.
</p>
<div class="separator" />

<style>
  svg {
    max-width: 200px;
  }
</style>
