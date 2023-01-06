<script>
  import { onMount } from "svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";

  // offset for stroke-dashoffset to emulate movement
  let offset = 0;
  // weight that scales the input
  let weight = 1;
  let input = 1;

  onMount(() => {
    let interval = setInterval(() => {
      offset -= 20;
    }, 400);
    return () => {
      clearInterval(interval);
    };
  });
</script>

<svg version="1.1" viewBox="0 0 240 25" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#000">
    <path
      id="input"
      stroke-dashoffset={offset}
      d="m5 12.5h85"
      stroke-dasharray="8, 8"
    />
    <path
      id="output"
      class={weight >= 0 ? "stroke-w4ai-blue" : "stroke-w4ai-red"}
      d="m150 12.5h90"
      stroke-dasharray="8, 8"
      stroke-dashoffset={offset}
      stroke-width={Math.abs(weight)}
    />
    <rect
      id="weight"
      x="95"
      y="5"
      width="50"
      height="15"
      class="fill-w4ai-yellow"
    />
  </g>
</svg>
<div class="flex justify-center items-center gap-3">
  <div><Latex>w</Latex></div>
  <Slider min={-5} max={5} bind:value={weight} step={0.1} />
</div>

<Table>
  <TableHead>
    <Row>
      <HeaderEntry>Variable</HeaderEntry>
      <HeaderEntry>Value</HeaderEntry>
    </Row>
  </TableHead>
  <TableBody>
    <Row>
      <DataEntry>
        Input <Latex>x</Latex>
      </DataEntry>
      <DataEntry>
        {input}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Weight <Latex>w</Latex>
      </DataEntry>
      <DataEntry>
        {weight}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Scaled Value <Latex>x * w</Latex>
      </DataEntry>
      <DataEntry>
        {weight * input}
      </DataEntry>
    </Row>
  </TableBody>
</Table>
