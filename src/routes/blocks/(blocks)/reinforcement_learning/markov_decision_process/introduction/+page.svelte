<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>Markov Decision Process - World4AI</title>
  <meta
    name="description"
    content="In reinforcement learning the Markov decison process is the mathematical formalization of the agent environment interaction."
  />
</svelte:head>

<h1>Markov Decision Process</h1>
<div class="separator" />

<Container>
  <SvgContainer maxWidth="300px">
    <svg version="1.1" viewBox="0 0 500 350" xmlns="http://www.w3.org/2000/svg">
      <g stroke="#000">
        <g id="connections" class="stroke-black fill-none" stroke-width="1px">
          <path d="m250 34 100 130" />
          <path d="m250 34-100 120" />
          <path d="m150 154-110 160" />
          <path d="m150 164 100 150" />
          <path d="m350 164 100 150" />
          <path d="m350 164-100 150" />
        </g>
        <g
          id="actions"
          class="fill-blue-200 stroke-black"
          stroke-dasharray="1, 1"
          stroke-linecap="round"
        >
          <ellipse
            id="left-action-focus"
            cx="144.72"
            cy="157.52"
            rx="25"
            ry="25"
            opacity="0.7"
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
          class="fill-slate-800"
          stroke-linecap="round"
          stroke="none"
        >
          <ellipse
            id="top-state-focus"
            cx="249.5"
            cy="36.616"
            rx="35"
            ry="35"
            opacity="0.2"
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
            opacity="0.2"
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
    algorithms to solve the problem. In reinforcement learning the tool that is
    used for this purpos is the <Highlight
      >Markov Decision Process</Highlight
    >, often abbrevieated as MDP.
    Many of the components of the Markov decision process were already covered
    in the previous chapter, but while the focus of the previous chapter was the
    intuition, this chapter is going to develop the
    necessary mathematical foundation.
  </p>
</Container>
<div class="separator" />
