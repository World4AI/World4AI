<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Alert from '$lib/Alert.svelte' 
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | MDP Introduction</title>
  <meta
    name="description"
    content="In reinforcement learning the Markov decison process is the mathematical formalization of the agent environment interaction."
  />
</svelte:head>

<Alert type='danger'>
  The chapter is in the process of being refactored.
</Alert>

<h1>Markov Decision Process</h1>
<div class="separator" />

<Container>
  <SvgContainer maxWidth="300px">
    <svg version="1.1" viewBox="0 0 500 350" xmlns="http://www.w3.org/2000/svg">
      <g stroke="#000">
        <g
          id="connections"
          fill="none"
          stroke="var(--text-color)"
          stroke-width="1px"
        >
          <path d="m250 34 100 130" />
          <path d="m250 34-100 120" />
          <path d="m150 154-110 160" />
          <path d="m150 164 100 150" />
          <path d="m350 164 100 150" />
          <path d="m350 164-100 150" />
        </g>
        <g
          id="actions"
          stroke="var(--text-color)"
          fill="var(--background-color)"
          stroke-dasharray="1, 1"
          stroke-linecap="round"
        >
          <ellipse
            id="left-action-focus"
            cx="144.72"
            cy="157.52"
            rx="25"
            ry="25"
          />
          <ellipse
            id="left-action"
            cx="144.72"
            cy="157.52"
            rx="16.504"
            ry="16.504"
          />
          <ellipse
            id="right-action"
            cx="349.98"
            cy="160.66"
            rx="16.504"
            ry="16.504"
          />
        </g>
        <g
          id="states"
          fill="var(--text-color)"
          stroke-dasharray="0.999998, 0.999998"
          stroke-linecap="round"
          stroke="none"
        >
          <ellipse
            id="top-state-focus"
            cx="249.5"
            cy="36.616"
            rx="35"
            ry="35"
            opacity="0.1"
          />
          <ellipse id="top-state" cx="249.5" cy="36.616" rx="24.5" ry="24.5" />
          <ellipse
            id="left-state"
            cx="44.173"
            cy="313.73"
            rx="24.5"
            ry="24.5"
          />
          <ellipse
            id="mid-state-focus"
            cx="248.06"
            cy="313.73"
            rx="35"
            ry="35"
            opacity="0.1"
          />
          <ellipse id="mid-state" cx="248.06" cy="313.73" rx="24.5" ry="24.5" />
          <ellipse
            id="right-state"
            cx="448.45"
            cy="313.73"
            rx="24.5"
            ry="24.5"
          />
        </g>
      </g>
    </svg>
  </SvgContainer>
  <p>
    In order to find an optimal solution to a reinforcement learning problem it
    is essential to formalize the problem in a mathematical framework. This
    allows researchers to study the properties of the problem and to develop
    algorithms. In reinforcement learning the tool that is commonly used for
    those purposes is the Markov decision process (often abbreviated as MDP).
  </p>
  <p>
    Many of the components of the Markov decision process were already covered
    in the previous chaptes, but while the focus of the previous chapter was the
    intuition of reinforcement learning this chapter is going to develop the
    necessary mathematical foundation.
  </p>
  <p>
    There are several ways to introduce MDPs, each way has it's own advantages
    and a different level of mathematical rigour. We are going to look at three
    different viewpoints of MDPs, while increasing the mathematical complexity
    one step at a time. At the end of the chapter we will have a look at what it
    means to find a solution to a Markov decision process.
  </p>
</Container>
<div class="separator" />
