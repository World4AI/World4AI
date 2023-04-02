<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Circle from "$lib/diagram/Circle.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  export let config;

  let width = 300;
  let height = config.type === "decision" ? 300 : 200;
  let stateRadius = 30;
  let stateY = height - stateRadius - (config.type !== "chain" ? 50 : 0);

  let stateDistance = (width - 40 - 2) / (config.states.length - 1);
  let actionCoordinates = [
    { x: 65, y: height / 2 - 50 },
    { x: 65 + 2 * stateDistance, y: height / 2 - 50 },
  ];
  let stateCoordinates = config.states.map((state, idx) => {
    return { x: 21 + idx * stateDistance, y: stateY };
  });
</script>

<SvgContainer maxWidth={"300px"}>
  <svg viewBox="0 0 {width} {height}">
    <!-- different connections -->
    {#if config.type !== "decision"}
      <Arrow
        data={[
          { x: 150, y: 20 },
          { x: 20, y: stateY },
        ]}
        strokeWidth="3"
        dashed={true}
        strokeDashArray="6 6"
        moving={true}
        showMarker={false}
        speed={50}
      />
      <Arrow
        data={[
          { x: 150, y: 20 },
          { x: 280, y: stateY },
        ]}
        strokeWidth="3"
        dashed={true}
        strokeDashArray="6 6"
        moving={true}
        showMarker={false}
        speed={50}
      />
    {:else}
      {#each actionCoordinates as coordinate, idx}
        <!--root to actions -->
        <Arrow
          data={[
            { x: 150, y: 20 },
            { x: coordinate.x, y: coordinate.y },
          ]}
          strokeWidth="3"
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
          showMarker={false}
          speed={50}
        />
        <!--actions to states/rewards -->
        <Arrow
          data={[
            { x: coordinate.x, y: coordinate.y },
            { x: stateCoordinates[idx * 2].x, y: stateY },
          ]}
          strokeWidth="3"
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
          showMarker={false}
          speed={50}
        />
        <Arrow
          data={[
            { x: coordinate.x, y: coordinate.y },
            { x: stateCoordinates[idx * 2 + 1].x, y: stateY },
          ]}
          strokeWidth="3"
          dashed={true}
          strokeDashArray="6 6"
          moving={true}
          showMarker={false}
          speed={50}
        />
      {/each}
    {/if}
    <!-- root -->
    <Circle
      x={150}
      y={20}
      r={18}
      text={config.root}
      type="latex"
      fontSize={20}
      class="fill-slate-400"
    />
    <!-- states -->
    {#each config.states as state, idx}
      <Circle
        x={stateCoordinates[idx].x}
        y={stateCoordinates[idx].y}
        r={20}
        text={state}
        fontSize={25}
        class="fill-red-400"
      />
    {/each}
    <!-- actions -->
    {#if config.type === "decision"}
      {#each config.actions as action, idx}
        <Circle
          x={actionCoordinates[idx].x}
          y={actionCoordinates[idx].y}
          r={20}
          text={action}
          fontSize={25}
          class="fill-purple-100"
        />
      {/each}
    {/if}
    <!-- rewards -->
    {#if config.type !== "chain"}
      {#each config.rewards as state, idx}
        <Circle
          x={stateCoordinates[idx].x}
          y={stateCoordinates[idx].y + 50}
          r={20}
          text={state}
          fontSize={25}
          class="fill-green-100"
        />
      {/each}
    {/if}
    <!-- probabilities -->
    {#each config.p as p, idx}
      <Block
        x={stateCoordinates[idx].x}
        y={stateCoordinates[idx].y - 50}
        width={40}
        height={30}
        text={p}
        type="latex"
        class="fill-blue-200"
        fontSize={18}
      />
    {/each}
    {#if config.type === "decision"}
      {#each config.actionp as p, idx}
        <Block
          x={actionCoordinates[idx].x}
          y={actionCoordinates[idx].y - 50}
          width={40}
          height={30}
          text={p}
          type="latex"
          class="fill-blue-200"
          fontSize={18}
        />
      {/each}
    {/if}
  </svg>
</SvgContainer>
