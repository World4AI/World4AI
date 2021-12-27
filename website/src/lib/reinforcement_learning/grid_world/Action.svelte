<script>
  import { tweened } from "svelte/motion";
  import { cubicOut } from "svelte/easing";

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

<svg width={size} height={size}>
  <circle
    cx={size / 2}
    cy={size / 2}
    r={size / 2 - 5}
    fill="none"
    stroke="var(--text-color)"
  />
  <defs>
    <marker
      id="arrowhead"
      markerWidth="10"
      markerHeight="7"
      refX="0"
      refY="3.5"
      orient="auto"
      fill="var(--text-color)"
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
      stroke="var(--text-color)"
      stroke-width="2"
      marker-end="url(#arrowhead)"
    />
  {/if}
</svg>
