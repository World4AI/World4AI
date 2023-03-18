<script>
  import { tweened } from "svelte/motion";
  import { cubicOut } from "svelte/easing";
  import SvgContainer from "$lib/SvgContainer.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  const rotation = tweened(0, {
    duration: 200,
    easing: cubicOut,
  });

  export let action;
  export let size = 150;

  $: {
    if (action !== null && action !== undefined) {
      rotation.set(actionToDegreeMapping[action]);
    }
  }
  let actionToDegreeMapping = {
    0: 270,
    1: 0,
    2: 90,
    3: 180,
  };
  let x1 = size * 0.2;
  let y1 = size / 2;
  let x2 = size * 0.8;
  let y2 = size / 2;
</script>

<SvgContainer maxWidth="100px">
  <svg viewBox="0 0 150 150">
    <circle
      cx={size / 2}
      cy={size / 2}
      r={size / 2 - 5}
      fill="none"
      stroke="black"
      class="fill-slate-300"
    />
    <defs>
      <marker
        id="arrowhead"
        markerWidth="10"
        markerHeight="7"
        refX="0"
        refY="3.5"
        orient="auto"
        fill="black"
      >
        <polygon points="0 0, 10 3.5, 0 7" />
      </marker>
    </defs>
    {#if action !== null}
      <line
        {x1}
        {y1}
        {x2}
        {y2}
        transform="rotate({$rotation}, {size / 2}, {size / 2})"
        stroke="black"
        stroke-width="2"
        marker-end="url(#arrowhead)"
      />
    {/if}
  </svg>
</SvgContainer>

<Table>
  <TableHead>
    <Row>
      <HeaderEntry>Action</HeaderEntry>
    </Row>
  </TableHead>
  <TableBody>
    <Row>
      <DataEntry>
        <span class="bg-red-100 px-5 py-1 rounded-full">
          {action}
        </span>
      </DataEntry>
    </Row>
  </TableBody>
</Table>
