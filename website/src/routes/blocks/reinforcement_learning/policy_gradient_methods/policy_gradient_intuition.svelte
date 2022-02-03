<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";
  import MDPTree from "$lib/reinforcement_learning/MDPTree.svelte";
  import { onMount } from "svelte";

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
  let maxLevels = 5;
  //the acvive path of the tree
  let active = null;

  let state1ProbLeft = 50;
  let state2ProbLeft = 50;

  $: policy = {
    0: [
      { action: 0, probability: 0.5 },
      { action: 1, probability: 0.5 },
    ],
    1: [
      { action: 0, probability: state1ProbLeft / 100 },
      { action: 1, probability: (100 - state1ProbLeft) / 100 },
    ],
    2: [
      { action: 0, probability: state2ProbLeft / 100 },
      { action: 1, probability: (100 - state2ProbLeft) / 100 },
    ],
    3: [
      { action: 0, probability: 0.5 },
      { action: 1, probability: 0.5 },
    ],
  };

  let rewards = { 0: -10, 1: -1, 2: -1, 3: 10 };

  let model = {
    0: {
      0: [{ nextState: 0, probability: 1 }],
      1: [{ nextState: 0, probability: 1 }],
    },
    1: {
      0: [
        { nextState: 0, probability: 0.8, terminal: true },
        { nextState: 1, probability: 0.2 },
      ],
      1: [
        { nextState: 1, probability: 0.2 },
        { nextState: 2, probability: 0.8 },
      ],
    },
    2: {
      0: [
        { nextState: 1, probability: 0.8 },
        { nextState: 2, probability: 0.2 },
      ],
      1: [
        { nextState: 2, probability: 0.2 },
        { nextState: 3, probability: 0.8, terminal: true },
      ],
    },
    3: {
      0: [{ nextState: 3, probability: 1 }],
      1: [{ nextState: 3, probability: 1 }],
    },
  };

  // build the markov decision model in a tree structure using recursion
  function buildTree() {
    let startTree = {
      level: 1,
      type: "state",
      value: startState,
      allRewards: [],
      allProbabilities: [],
      prevExpectation: 1,
      children: [],
    };
    // the variable is used to determine which path a node belongs two
    // for example the top most node is part of all paths
    let pathNum = 0;
    let productReducer = (prev, current) => prev * current;
    let sumReducer = (prev, current) => prev + current;
    function createTree(maxLevel, node) {
      if (node.level >= maxLevel || node.terminal) {
        // remember the path number
        pathNum += 1;
        if (!node.paths) {
          node.paths = [];
        }
        node.paths.push(pathNum);

        //calcualate the expectation contribution of the path
        let pathProb = node.allProbabilities.reduce(productReducer, 1);
        let pathReturn = node.allRewards.reduce(sumReducer, 0);
        node.expectation = pathProb * pathReturn;

        return [pathNum];
      }
      let pathsNumbers = [];
      let expectation = 0;
      let childNode = {};
      if (node.type === "state") {
        let actions = policy[node.value];
        actions.forEach((action) => {
          childNode = {
            level: node.level + 1,
            type: "action",
            value: action.action,
            prevValue: node.value,
            allRewards: [...node.allRewards],
            allProbabilities: [...node.allProbabilities, action.probability],
            probability: action.probability,
            marked: false,
            children: [],
          };
          node.children.push(childNode);
          let path = createTree(maxLevel, childNode);
          expectation += childNode.expectation;
          path.flat().forEach((num) => {
            pathsNumbers.push(num);
          });
        });
      } else if (node.type === "action") {
        let nextStates = model[node.prevValue][node.value];
        nextStates.forEach((nextState) => {
          childNode = {
            level: node.level + 1,
            type: "state",
            reward: rewards[nextState.nextState],
            value: nextState.nextState,
            prevValue: node.value,
            allRewards: [...node.allRewards, rewards[nextState.nextState]],
            allProbabilities: [...node.allProbabilities, nextState.probability],
            probability: nextState.probability,
            terminal: nextState.terminal,
            marked: false,
            children: [],
          };
          node.children.push(childNode);
          let path = createTree(maxLevel, childNode);
          expectation += childNode.expectation;
          path.flat().forEach((num) => {
            pathsNumbers.push(num);
          });
        });
      }

      if (!node.paths) {
        node.paths = [];
      }
      pathsNumbers.forEach((num) => {
        node.paths.push(num);
      });
      node.expectation = expectation;
      return pathsNumbers;
    }
    createTree(maxLevels, startTree);
    return startTree;
  }

  let tree;
  let mounted = false;
  onMount(() => {
    tree = buildTree();
    mounted = true;
  });

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
  or right. The initial state is state 1 and the goal of the environment is to
  reach state the terminal state 3, where the agent receives a reward of 10.
  Each timestep the agent takes the agent receives a reward of -1. If the agent
  lands in the terminal state 0, the agent receives a reward of -10. To simplify
  the coming calculations we are going to let the agent take up to 2 actions.
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

<p>
  Below is depicted the model of the environment. In 80% of cases the
  environment transitions into the chosen direction, where 0 is the left and 1
  is the right action. In 20% of cases the state does not change. Once the agent
  reaches state 0 or 1, the reward is exhausted and the agent can not leave the
  terminal state.
</p>
<!-- Show model-->
<Table header={modelHeader} data={modelData} />
<p>
  The optimal policy is obviously to keep moving right, but we initialize the
  agent with random behaviour in order to demonstrate the learning process.
</p>
<!-- Show policy -->
<Table header={policyHeader} data={policyData} />

<p>
  Below is the tree representing all paths that can be achieved after two
  actions when starting in state 1. The larger circles are the individual states
  with the corresponding rewards. The smaller cirlces are the possible actions,
  where the left circles represent the movement to the left and vice versa.
  Below the large state circles is a little map that indicates the location of
  the agent. The two components that you should notice are the red and blue
  numbers. The red numbers are the probabilities to take a certain action in a
  state. These are the numbers we actually have control over and would like to
  optimize. The blue numbers are the expectated returns. The expected return
  under the top node is the overall expected return from the starting state. All
  the other blue values are the corresponding contributions to the expected
  return. Essentially each individual blue number is the sum of the blue numbers
  below and the expected values at the bottom are the expectations for that
  particular path, that we can calculate if we multiply the probability of the
  path with the sum of rewards in that path. The white numbers outside the
  circles are the probabilities to transition into the next state. These numbers
  we can not influence.
</p>
<!-- Show model tree -->
{#if mounted}
  <MDPTree root={tree} {active} />
{/if}
<label>
  {state1ProbLeft}
  <input type="range" bind:value={state1ProbLeft} min="0" max="100" />
</label>
<label>
  {state2ProbLeft}
  <input type="range" bind:value={state2ProbLeft} min="0" max="100" />
</label>
<p>
  The goal of reinforcement learning is to maximize the expected return, we have
  therefore to maximize the upper blue value in the drawing above.
</p>
<div class="separator" />

<style>
  svg {
    max-width: 700px;
  }
</style>
