<script>
  // can be info, warning and danger
  export let type;
  let background;

  if (type === "info") {
    background = "bg-w4ai-lightblue";
  } else if (type === "warning") {
    background = "bg-w4ai-yellow";
  } else if (type === "danger") {
    background = "bg-w4ai-red";
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
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="feather feather-info"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
      <p class="m-0">Info</p>
    {:else if type === "warning"}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="feather feather-alert-triangle"
      >
        <path
          d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"
        />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
      <p class="m-0">Warning</p>
    {:else if type === "danger"}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="feather feather-alert-octagon"
      >
        <polygon
          points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"
        />
        <line x1="12" y1="8" x2="12" y2="12" /><line
          x1="12"
          y1="16"
          x2="12.01"
          y2="16"
        />
      </svg>
      <p class="m-0">Danger</p>
    {/if}
  </div>
  <p class="m-0">
    <slot />
  </p>
</div>
