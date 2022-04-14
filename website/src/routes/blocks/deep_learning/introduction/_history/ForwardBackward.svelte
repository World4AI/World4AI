<script>
  import { onMount } from "svelte";

  export let type = "forward";
  let offset = 0;

  onMount(() => {
    let interval = setInterval(() => {
      if (type === "forward") {
        offset += 20;
      } else if (type === "backward") {
        offset -= 20;
      }
    }, 300);
    return () => {
      clearInterval(interval);
    };
  });
</script>

<div class="container">
  <div class="svg-container">
    <svg version="1.1" viewBox="0 0 350 200" xmlns="http://www.w3.org/2000/svg">
      <g fill="none" stroke="var(--text-color)">
        <g
          id="connections"
          stroke={type === "forward"
            ? "var(--main-color-2)"
            : "var(--main-color-1)"}
          stroke-dashoffset={offset}
          stroke-dasharray="4, 8"
        >
          <path d="m65 35h95" />
          <path d="m65 50 95 100" />
          <path d="m65 150 95-100" />
          <path d="m65 165h95" />
          <path d="m190 35h95" />
          <path d="m190 50 95 100" />
          <path d="m190 150 95-100" />
          <path d="m190 165h95" />
        </g>
        <g id="neurons">
          <rect x="35" y="20" width="30" height="30" />
          <rect x="35" y="150" width="30" height="30" />
          <rect x="160" y="20" width="30" height="30" />
          <rect x="160" y="150" width="30" height="30" />
          <rect x="285" y="20" width="30" height="30" />
          <rect x="285" y="150" width="30" height="30" />
        </g>
      </g>
    </svg>
  </div>
</div>

<style>
  .svg-container {
    max-width: 500px;
    margin: 0 auto;
  }
</style>
