<script>
  import { Info, AlertTriangle, AlertOctagon, Quote } from "lucide-svelte";
  // can be info, warning and danger
  export let type;
  let background;

  if (type === "info") {
    background = "bg-w4ai-lightblue";
  } else if (type === "warning") {
    background = "bg-w4ai-yellow";
  } else if (type === "danger") {
    background = "bg-w4ai-red";
  } else if (type === "quote") {
    background = "bg-gray-200";
  }

  let color = type === "danger" ? "white" : "black";
</script>

<div
  class={`flex flex-col justify-start items-start gap-3 px-6 py-4 my-2 ${background}`}
  style:color
>
  <div
    class="flex justify-start items-center gap-2 border-b border-black/5 w-full pb-2"
  >
    <!-- we use feather icons -->
    {#if type === "info"}
      <Info />
      <p class="m-0">Info</p>
    {:else if type === "warning"}
      <AlertTriangle />
      <p class="m-0">Warning</p>
    {:else if type === "danger"}
      <AlertOctagon />
      <p class="m-0">Danger</p>
    {:else if type === "quote"}
      <Quote />
      <p class="m-0">Quote</p>
    {/if}
  </div>
  <p class="m-0">
    <slot />
  </p>
</div>
