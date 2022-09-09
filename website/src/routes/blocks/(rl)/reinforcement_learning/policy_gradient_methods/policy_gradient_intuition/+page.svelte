<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Table from "$lib/Table.svelte";
  import MDPTree from "$lib/reinforcement_learning/MDPTree.svelte";
  import Slider from "$lib/Slider.svelte";

  import { onMount } from "svelte";

  //the acvive path of the tree
  let active = null;
  onMount(() => {
    let interval = setInterval(() => {
      active = Math.ceil(Math.random() * 13);
    }, 1000);

    return () => clearInterval(interval);
  });

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

  let state1ProbRight = 50;
  let state2ProbRight = 50;

  $: policy = {
    0: [
      { action: 0, probability: 0.5 },
      { action: 1, probability: 0.5 },
    ],
    1: [
      { action: 0, probability: (100 - state1ProbRight) / 100 },
      { action: 1, probability: state1ProbRight / 100 },
    ],
    2: [
      { action: 0, probability: (100 - state2ProbRight) / 100 },
      { action: 1, probability: state2ProbRight / 100 },
    ],
    3: [
      { action: 0, probability: 0.5 },
      { action: 1, probability: 0.5 },
    ],
  };

  let constantPolicy = {
    0: [
      { action: 0, probability: 0.5 },
      { action: 1, probability: 0.5 },
    ],
    1: [
      { action: 0, probability: 0.5 },
      { action: 1, probability: 0.5 },
    ],
    2: [
      { action: 0, probability: 0.5 },
      { action: 1, probability: 0.5 },
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
  function buildTree(policy, model, rewards) {
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
  $: if (policy) {
    tree = buildTree(policy, model, rewards);
  }

  let constantTree = buildTree(constantPolicy, model, rewards);
  let markedTree1 = buildTree(constantPolicy, model, rewards);
  let markedTree2 = buildTree(constantPolicy, model, rewards);

  function markTree(node, level) {
    if (node.level === level) {
      node.marked = true;
    }

    node.children.forEach((childNode) => {
      markTree(childNode, level);
    });
  }

  markTree(markedTree1, 2);
  markTree(markedTree2, 4);

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

<svelte:head>
  <title>World4AI | Reinforcement Learning | Policy Gradient Intuition</title>
  <meta
    name="description"
    content="Policy gradient methods change the policy of the agent by sampling returns and scaling the probabilities of actions by those returns."
  />
</svelte:head>

<h1>Policy Gradient Intuition</h1>
<div class="separator" />

<Container>
  <p>
    The derivation and the understanding of the policy gradient theorem is
    mathematically quite involved and can be relatively intimidating. Chances
    are you will need to go through the derivations several times to fully grasp
    the concepts. To facilitate the understanding we will start the policy
    gradient chapter with a section that will build some intuition.
  </p>

  <p>
    For that purpose we will use a gridworld, where the agent can only move left
    or right. The initial state is state 1 and the goal of the environment is to
    reach the terminal state 3, where the agent receives a reward of 10. Each
    timestep the agent takes the agent receives a reward of -1. If the agent
    lands in the terminal state 0, the agent receives a reward of -10. To
    simplify the coming calculations we are going to let the agent take up to 2
    actions. After the 2 actions the game will restart.
  </p>
  <!-- Show the grid world -->

  <SvgContainer maxWidth="500px">
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
  </SvgContainer>

  <p>
    In the table below we can observe the model of the environment. The model
    depicts the probability of transitioning given the state, the action and the
    next state. In 80% of cases the environment transitions into the chosen
    direction, where 0 is the left and 1 is the right action. In 20% of cases
    the state does not change. Once the agent reaches state 0 or 1, the reward
    is exhausted and the agent can not leave the terminal state.
  </p>
  <!-- Show model-->
  <Table header={modelHeader} data={modelData} />
  <p>
    The optimal policy is obviously to keep moving right, but we initialize the
    policy of the agent with random behaviour in order to demonstrate the
    learning process.
  </p>
  <!-- Show policy -->
  <Table header={policyHeader} data={policyData} />
  <p>
    Below is the tree representing all paths that can be reached after two
    actions from state 1. The larger circles are the states with the
    corresponding rewards. Below those circles you can see a mini map, where the
    highlighted square indicates the location of the agent. The smaller cirlces
    are the possible actions, where the left circles represent the movement to
    the left and vice versa. The two components that you should notice are the
    red and blue numbers. The red numbers are the probabilities to take a
    certain action in a state. These are the numbers we actually have control
    over and would like to optimize. The blue numbers are the expectated
    returns. The expected return under the top node is the overall expected
    return from the starting state. All the other blue values are the
    corresponding contributions to the expected return. Essentially each
    individual blue number is the sum of the blue numbers below and the expected
    values at the bottom are the expectations for that particular path, that we
    can calculate if we multiply the probability of the path with the sum of
    rewards in that path. The white numbers outside the circles are the
    probabilities to transition into the next state. These numbers we can not
    influence.
  </p>
  <!-- Show not changing tree -->
  <MDPTree root={constantTree} />

  <!-- Show marked tree -->
  <p>
    We have already noticed, that it would be optimal to go to the right all the
    time, but how can we use policy gradient methods for that purpose? Let us
    split the problem into two halves at state 1. How do we tweak the
    probabilities to go left and right at the top node in the tree below? The
    left path produces an expected value of -4.56 and the right path produces
    the expected value of 0.4. If we change the probability to go into a
    particular direction proportional to the expected values, then we would
    automatically increase the probability to move right and decrease the
    probability to move left.
  </p>
  <MDPTree root={markedTree1} />
  <p>
    We take the same approach for each sublevel of the tree. For example the two
    halves in the right corner depict the state two. The expected value of going
    left is -0.4 and of going right is 1.36, we therefore increase the
    probability of going right and decrease the probability of going left.
  </p>
  <MDPTree root={markedTree2} />
  <!-- Show tree with sliders -->
  <p>
    Below is the tree with the same initial conditions. We provided two sliders,
    that allow you to tweak the probabilities to go left and right in state 1
    and 2. When you move the sliders to the right you increase the probability
    to go right and because the probabilities sum up to 1, you simultaneously
    decrease the probability of going to the left. You should also observe that
    state 1 is encountered several times, therefore there is a lot of
    interaction by tweaking the above slider.
  </p>
  <MDPTree root={tree} />
  <div class="input-group">
    <label for="state1"
      >State 1: Probability left {100 - state1ProbRight}% | Probability right {state1ProbRight}%</label
    >
    <Slider min="0" max="100" bind:value={state1ProbRight} />
  </div>
  <div class="input-group">
    <label for="state2"
      >State 2: Probability left {100 - state2ProbRight}% | Probability right {state2ProbRight}%</label
    >
    <Slider min="0" max="100" bind:value={state2ProbRight} />
  </div>
  <p>
    Lastly you probably already noticed that in order to calculate the expected
    value of each of the subtrees you would need access to the model of the
    environment. Well it turns out that according to the policy gradient theorem
    you can sample the different paths, as indicated in the animation below.
    After you collected some samples you can scale the probability of actions by
    the returns of those samples. Some individual draws might change the
    probability into the wrong direction but in expectation we should find the
    optimal policy.
  </p>
  <MDPTree root={constantTree} {active} />
  <p>
    In practice we deal with policies that are represented by a neural network,
    but the intuition should remain the same. We can change the probabilities of
    the neural network by tweaking its weights. We use a machine learning
    framework like PyTorch or TensorFlow to calculate the gradient of the
    policy. That way we can figure out how we should tweak the weights to
    increase the probability of an action. And finally we scale the gradient by
    sampled returns. Larger returns increase the probabilities more than lower
    returns (or negative returns that decrease the probabilities). Because the
    probabilities sum to 1 good paths should become more likely to be drawn and
    bad paths will become unlikely.
  </p>
  <div class="separator" />
</Container>
