<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";

  //Grid configuration
  let gridWidth = 350;
  let gridHeight = 60;
  let gridBoxSize = 50;
  let gridHeightPadding = 20;

  let gridConfig = [
    { state: 0, type: "Terminal", reward: -10, color: "var(--main-color-1)" },
    { state: 1, type: "Start", reward: -1, color: "var(--main-color-1)" },
    { state: 2, reward: -1, color: "var(--main-color-1)" },
    { state: 3, type: "Terminal", reward: 10, color: "var(--main-color-2)" },
  ];

  let gridPiece = gridWidth / gridConfig.length;

  // tree configuration
  let startState = 1;
  let treeWidth = 200;
  let treeHeight = 200;
  let stateRadius = 8;
  let actionRadius = 5;

  let policy = {
    0: [0.5, 0.5],
    1: [0.5, 0.5],
    2: [0.5, 0.5],
    3: [0.5, 0.5],
  };

  let model = {
    0: { 0: [[0, 1]], 1: [[0, 1]] },
    1: {
      0: [
        [0, 0.8],
        [1, 0.2],
      ],
      1: [
        [1, 0.2],
        [2, 0.8],
      ],
    },
    2: {
      0: [
        [1, 0.8],
        [2, 0.2],
      ],
      1: [
        [2, 0.2],
        [3, 0.8],
      ],
    },
    3: { 0: [[3, 1]], 1: [[3, 1]] },
  };

  // path variable to determine the flow of the path
  let tree = [
    {
      id: 1,
      layer: 1,
      type: "state",
      value: 1,
      children: [11, 12],
    },
    { id: 11, layer: 2, type: "action", value: 0, children: [21, 22] },
    { id: 12, layer: 2, type: "action", value: 1, children: [23, 24] },
    { id: 21, layer: 3, type: "state", value: 0, children: [] },
    { id: 22, layer: 3, type: "state", value: 1, children: [] },
    { id: 23, layer: 3, type: "state", value: 1, children: [] },
    { id: 24, layer: 3, type: "state", value: 2, children: [] },
    { id: 31, layer: 4, type: "action", value: 0, children: [21, 22] },
    { id: 32, layer: 4, type: "action", value: 1, children: [23, 24] },
    { id: 33, layer: 4, type: "action", value: 0, children: [21, 22] },
    { id: 34, layer: 4, type: "action", value: 1, children: [23, 24] },
  ];

  // data for tables
  let modelHeader = [
    "State",
    "Action",
    "Next State",
    "Probability",
    "Reward",
    "Terminal",
  ];
  let modelData = [
    [0, 0, 0, 1.0, 0, true],
    [0, 1, 0, 1.0, 0, true],
    [1, 0, 0, 0.8, -10, true],
    [1, 0, 1, 0.2, -1, false],
    [1, 1, 1, 0.2, -1, false],
    [1, 1, 2, 0.8, -1, false],
    [2, 0, 1, 0.8, -1, false],
    [2, 0, 2, 0.2, -1, false],
    [2, 1, 2, 0.2, -1, false],
    [2, 1, 3, 0.8, 10, true],
    [3, 0, 3, 1, 0, true],
    [3, 1, 3, 1, 0, true],
  ];

  let policyHeader = ["State", "Probability Left (0)", "Probability Right (1)"];
  let policyData = [
    [0, 0.5, 0.5],
    [1, 0.5, 0.5],
    [2, 0.5, 0.5],
    [3, 0.5, 0.5],
  ];
</script>

<h1>Policy Gradient Intuition</h1>
<Question
  >Can you explain the policy gradient algorithm in an intuive way?</Question
>
<div class="separator" />

<p>
  The derivation and the understanding of the policy gradient theorem is
  mathematically quite involved and can be relatively intimidating. Chances are
  you will need to go through the derivations several times to fully grasp the
  concepts. To facilitate the understanding we will start the policy gradient
  chapter with a section that will build some intuition.
</p>

<p>
  For that purpose we will use a gridworld, where the agent can only move left
  or right.
</p>
<!-- Show the grid world -->
<svg viewBox="0 0 {gridWidth} {gridHeight + 2 * gridHeightPadding}">
  {#each gridConfig as cell, idx}
    <rect
      x={gridPiece * idx + gridPiece / 2 - gridBoxSize / 2}
      y={gridHeightPadding + gridHeight / 2 - gridBoxSize / 2}
      width={gridBoxSize}
      height={gridBoxSize}
      fill={cell.color}
      stroke="black"
    />
    <text
      stroke="none"
      fill="black"
      x={gridPiece * idx + gridPiece / 2}
      y={gridHeightPadding + gridHeight / 2}
      dominant-baseline="middle"
      text-anchor="middle">{cell.reward}</text
    >
    {#if cell.type}
      <text
        stroke="none"
        font-size="10"
        fill="var(--text-color)"
        x={gridPiece * idx + gridPiece / 2}
        y={gridHeightPadding / 2}
        dominant-baseline="middle"
        text-anchor="middle">{cell.type}</text
      >
    {/if}
    <text
      stroke="none"
      font-size="10"
      fill="var(--text-color)"
      x={gridPiece * idx + gridPiece / 2}
      y={gridHeightPadding + gridHeight + gridHeightPadding / 2}
      dominant-baseline="middle"
      text-anchor="middle">State: {cell.state}</text
    >
  {/each}
</svg>

<!-- Show model and policy -->
<Table header={modelHeader} data={modelData} />
<Table header={policyHeader} data={policyData} />

<!-- Show model tree -->
<style>
  svg {
    border: 1px solid black;
    max-width: 700px;
  }
</style>
