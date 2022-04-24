<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import { onMount } from "svelte";

  let offsetTop = 0;
  let offsetBottom = 0;

  onMount(() => {
    let interval = setInterval(() => {
      offsetTop += 20;
      offsetBottom -= 20;
    }, 300);

    return () => clearInterval(interval);
  });
</script>

<SvgContainer maxWidth={"500px"}>
  <svg version="1.1" viewBox="0 0 500 120" xmlns="http://www.w3.org/2000/svg">
    <g fill="none" stroke="var(--text-color)">
      <g id="environment">
        <rect
          x=".59406"
          y=".59406"
          width="118.81"
          height="118.81"
          opacity=".999"
          stroke-linejoin="round"
          stroke-width="1.1881"
        />
        <rect
          x="10.495"
          y="10.495"
          width="99.01"
          height="99.01"
          opacity=".999"
          stroke-linejoin="round"
          stroke-width="1"
        />
        <g stroke-width="1">
          <path d="m20.396 10.495v99.01" />
          <path d="m30.297 10.495v99.01" />
          <path d="m40.198 10.495v99.01" />
          <path d="m50.099 10.495v99.01" />
          <path d="m60 10.495v99.01" />
          <path d="m69.901 10.495v99.01" />
          <path d="m79.802 10.495v99.01" />
          <path d="m89.703 10.495v99.01" />
          <path d="m99.604 10.495v99.01" />
          <path d="m10.495 20.396h99.01" />
          <path d="m10.495 30.297h99.01" />
          <path d="m10.495 40.198h99.01" />
          <path d="m10.495 50.099h99.01" />
          <path d="m10.495 60h99.01" />
          <path d="m10.495 69.901h99.01" />
          <path d="m10.495 79.802h99.01" />
          <path d="m10.495 89.703h99.01" />
          <path d="m10.495 99.604h99.01" />
        </g>
      </g>
      <g id="agent">
        <g>
          <rect
            x="409.54"
            y="28.153"
            width="63.797"
            height="63.797"
            ry="0"
            stroke-width="1"
          />
          <rect
            x="415.92"
            y="34.533"
            width="51.036"
            height="51.036"
            ry="0"
            stroke-width="1"
          />
          <circle
            cx="441.44"
            cy="60.05"
            r="9.6947"
            fill-rule="evenodd"
            stroke-linejoin="round"
            stroke-width="1"
          />
        </g>
        <g stroke-width="1">
          <path d="m420 26.667v-13.333h-13.333v-13.333" />
          <path d="m433.33 26.667v-26.667" />
          <path d="m446.67 26.667v-26.667" />
          <path d="m460 26.667v-13.333h13.333v-13.333" />
          <path d="m473.33 40h13.333v-13.333h13.333" />
          <path d="m473.33 53.333h26.667" />
          <path d="m473.33 66.667h26.667" />
          <path d="m473.33 80h13.333v13.333h13.333" />
          <path d="m406.67 40h-13.333v-13.333h-13.333" />
          <path d="m406.67 53.333h-26.667" />
          <path d="m406.67 66.667h-26.667" />
          <path d="m406.67 80h-13.333v13.333h-13.333" />
          <path d="m420 93.333v13.333h-13.333v13.333" />
          <path d="m433.33 93.333v26.667" />
          <path d="m446.67 93.333v26.667" />
          <path d="m460 93.333v13.333h13.333v13.333" />
        </g>
      </g>
      <g id="direction">
        <path
          d="m130 10h240"
          stroke-dasharray="8,2,1,2"
          stroke-dashoffset={offsetTop}
        />
        <path
          d="m370 110h-240"
          stroke-dasharray="8,2,1,2"
          stroke-dashoffset={offsetBottom}
        />
      </g>
    </g>
  </svg>
</SvgContainer>

