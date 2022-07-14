<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import { onMount } from "svelte";
  let offset = 0;

  onMount(() => {
    let interval = setInterval(() => {
      offset += 20;
    }, 1000);

    return () => clearInterval(interval);
  });
</script>

<SvgContainer maxWidth={"500px"}>
  <svg version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
    <g fill="none" stroke="var(--text-color)">
      <g stroke-width="1px">
        <path d="m0 24.242h500" />
        <path d="m500 175.76h-500" />
      </g>
      <g id="car" stroke-linejoin="round">
        <rect
          transform="rotate(90)"
          x="40.513"
          y="-451.64"
          width="23.368"
          height="80.823"
          ry="0"
        />
        <rect
          transform="rotate(90)"
          x="44.26"
          y="-438.68"
          width="15.873"
          height="54.901"
          ry="0"
          stroke-width=".67927"
        />
      </g>
      <path
        id="stripes"
        d="m-11.924 100h533.36"
        stroke-dasharray="40,80"
        stroke-dashoffset={offset}
        stroke-width="10"
      />
    </g>
  </svg>
</SvgContainer>

