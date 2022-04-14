<script>
  import { onMount } from "svelte";

  let offset = 0;
  onMount(() => {
    let interval = setInterval(() => {
      offset -= 20;
    }, 300);
    return () => {
      clearInterval(interval);
    };
  });
</script>

<svg version="1.1" viewBox="0 0 220 55" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="var(--text-color)">
    <rect id="neuron" x="105" y="17.5" width="30" height="30" />
    <g id="paths" stroke-dasharray="4, 2" stroke-dashoffset={offset}>
      <path d="m10 37.5h95" />
      <path d="m120 17.5v-10h-30v20h15" />
      <path d="m135 32.5h75" />
    </g>
  </g>
</svg>
