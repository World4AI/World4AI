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
    <g id="weights" class="fill-w4ai-yellow">
      <rect x="63" y="6.5" width="45" height="10" stroke-width={0.5} />
      <rect x="63" y="82.5" width="45" height="10" stroke-width={0.5} />
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
        class={sum >= 0 ? "stroke-w4ai-blue" : "stroke-w4ai-red"}
      />
      <path
        id="output-top"
        d="m111 10s60-5 60 40"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        class={weight1 >= 0 ? "stroke-w4ai-blue" : "stroke-w4ai-red"}
        stroke-width={Math.abs(weight1)}
      />
      <path
        id="output-bot"
        d="m111 90s60 5 60-40"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        class={weight2 >= 0 ? "stroke-w4ai-blue" : "stroke-w4ai-red"}
        stroke-width={Math.abs(weight2)}
      />
    </g>

    <g id="sum-sign">
      <rect
        class="fill-blue-200"
        x="165"
        y="43.695"
        width="12"
        height="12"
        ry="0"
        stroke-linejoin="bevel"
      />
      <path d="m171 45.7v8" stroke-width="1px" />
      <path d="m167 50h8" stroke-width="1px" />
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
