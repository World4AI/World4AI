<script>
  import { onMount } from "svelte";
  import Latex from "$lib/Latex.svelte";
  import Slider from "$lib/Slider.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";

  // offset for stroke-dashoffset to emulate movement
  let offset = 0;
  let weight1 = 1;
  let weight2 = 1;
  let input1 = 1;
  let input2 = 1;
  $: scaled1 = weight1 * input1;
  $: scaled2 = weight2 * input2;
  $: sum = scaled1 + scaled2;

  onMount(() => {
    let interval = setInterval(() => {
      offset -= 20;
    }, 300);
    return () => {
      clearInterval(interval);
    };
  });
</script>

<svg version="1.1" viewBox="0 0 240 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#000">
    <g id="sum-sign">
      <rect
        fill="var(--main-color-4)"
        x="141.37"
        y="43.695"
        width="12.609"
        height="12.609"
        ry="0"
        stroke-linejoin="bevel"
      />
      <path d="m147.67 45v10" stroke-width="1px" />
      <path d="m142.67 50h10" stroke-width="1px" />
    </g>
    <g id="weights" fill="var(--main-color-3)">
      <rect id="weight" x="61" y="5" width="50" height="15" />
      <rect x="61" y="80" width="50" height="15" />
    </g>
    <g id="flow">
      <path
        id="input-top"
        d="m5 12.5h55"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
      />
      <path
        id="input-bot"
        d="m5 87.5h55"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
      />
      <path
        id="summed-output"
        d="m175.64 49.82h63.71"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        stroke-width={Math.abs(sum)}
        stroke={sum >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
      />
      <path
        id="output-top"
        d="m111 10s60-5 60 40"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        stroke={weight1 >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
        stroke-width={Math.abs(weight1)}
      />
      <path
        id="output-bot"
        d="m111 90s60 5 60-40"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        stroke={weight2 >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
        stroke-width={Math.abs(weight2)}
      />
    </g>
  </g>
</svg>
<div class="flex justify-center items-center gap-2">
  <div><Latex>w_1</Latex></div>
  <Slider min={-5} max={5} step={0.1} bind:value={weight1} />
</div>
<div class="flex justify-center items-center gap-2">
  <div><Latex>w_2</Latex></div>
  <Slider min={-5} max={5} step={0.1} bind:value={weight2} />
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
        Input <Latex>x_1</Latex>
      </DataEntry>
      <DataEntry>
        {input1}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Weight <Latex>w_1</Latex>
      </DataEntry>
      <DataEntry>
        {weight1}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Scaled Value <Latex>x_1 * w_1</Latex>
      </DataEntry>
      <DataEntry>
        {weight1 * input1}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Input <Latex>x_2</Latex>
      </DataEntry>
      <DataEntry>
        {input2}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Weight <Latex>w_2</Latex>
      </DataEntry>
      <DataEntry>
        {weight2}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Scaled Value <Latex>x_2 * w_2</Latex>
      </DataEntry>
      <DataEntry>
        {weight1 * input1}
      </DataEntry>
    </Row>
    <Row>
      <DataEntry>
        Weighted Sum <Latex>\sum_j x_j w_j</Latex>
      </DataEntry>
      <DataEntry>
        {sum.toFixed(2)}
      </DataEntry>
    </Row>
  </TableBody>
</Table>
