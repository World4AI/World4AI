<script>
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import { getContext } from "svelte";

  export let root;
  export let level = 0;
  export let parentColumn = 0;

  let column = 0;
  let levels = getContext("levels");
  if (level in levels) {
    levels[level] += 1;
    column = levels[level];
  } else {
    levels[level] = 0;
  }

  //sizes
  let nameHeight = 30;
  let nameWidth = 120;
  let nodeSize = 60;
  let opSize = 30;

  // coordinates
  let nodeStartXName = 60;
  let nodeStartYName = 25;
  let nodeStartXData = 30;
  let nodeStartYData = 70;
  let nodeStartXGrad = 90;
  let nodeStartYGrad = 70;

  // offset per depths and width of tree
  let levelOffset = 200;
  let columnOffset = 150;
</script>

{#if level !== 0}
  <Arrow
    data={[
      {
        x: nodeStartXData + parentColumn * columnOffset + columnOffset / 2,
        y:
          nodeStartYData +
          (level - 1) * levelOffset +
          levelOffset / 2 +
          opSize / 2,
      },
      {
        x: nodeStartXData + column * columnOffset,
        y: nodeStartYData + level * levelOffset - nodeSize / 2,
      },
    ]}
    strokeWidth={2}
    showMarker={false}
    dashed={true}
    strokeDashArray={"8 8"}
  />
{/if}

<!-- name rectangle -->
<Block
  x={nodeStartXName + column * columnOffset}
  y={nodeStartYName + level * levelOffset}
  width={nameWidth}
  height={nameHeight}
  text={`${root._name}`}
  fontSize={16}
  color="var(--main-color-2)"
/>

<!-- data rectangle -->
<Block
  x={nodeStartXData + column * columnOffset}
  y={nodeStartYData + level * levelOffset}
  width={nodeSize}
  height={nodeSize}
  text={`${root.data.toFixed(2)}`}
  fontSize={16}
  color="var(--main-color-2)"
/>

<!--grad rectangle -->
<Block
  x={nodeStartXGrad + column * columnOffset}
  y={nodeStartYGrad + level * levelOffset}
  width={nodeSize}
  height={nodeSize}
  text={`${root.grad.toFixed(2)}`}
  fontSize={16}
  color="var(--main-color-1)"
/>

<!-- operation rectangle -->
{#if root._op}
  <Block
    x={nodeStartXData + column * columnOffset + columnOffset / 2}
    y={nodeStartYData + level * levelOffset + levelOffset / 2}
    width={opSize}
    height={opSize}
    text={root._op}
    fontSize={20}
    color="var(--main-color-3)"
  />
  <Arrow
    data={[
      {
        x: nodeStartXData + column * columnOffset,
        y: nodeStartYData + level * levelOffset + nodeSize / 2,
      },
      {
        x: nodeStartXData + column * columnOffset + columnOffset / 2,
        y: nodeStartYData + level * levelOffset + levelOffset / 2 - opSize / 2,
      },
    ]}
    strokeWidth={2}
    showMarker={false}
    dashed={true}
    strokeDashArray={"8 8"}
  />
{/if}

<!-- recursive calls to node-->
{#each root._prev as child}
  <svelte:self root={child} level={level + 1} parentColumn={column} />
{/each}
