<script>
  import { onMount } from "svelte";
  // offset for stroke-dashoffset to emulate movement
  let offset = 0;
  // weight that scales the input
  let weight = 1;

  onMount(() => {
    let interval = setInterval(() => {
      offset += 20;
    }, 300);
    return () => {
      clearInterval(interval);
    };
  });

  //functions to change the scaling factor
  function increaseWeight() {
    if (weight < 9) {
      weight += 1;
    }
  }
  function decreaseWeight() {
    if (weight > -9) {
      weight -= 1;
    }
  }
</script>

<svg version="1.1" viewBox="0 0 500 150" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="marker81625" overflow="visible" orient="auto">
      <path
        transform="scale(.4)"
        d="m5.77 0-8.65 5v-10z"
        fill="var(--background-color)"
        fill-rule="evenodd"
        stroke="var(--background-color)"
        stroke-width="1pt"
      />
    </marker>
    <marker id="TriangleOutM" overflow="visible" orient="auto">
      <path
        transform="scale(.4)"
        d="m5.77 0-8.65 5v-10z"
        fill="var(--background-color)"
        fill-rule="evenodd"
        stroke="var(--background-color)"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <rect
    x="5"
    y="45"
    width="60"
    height="60"
    fill="none"
    stroke="var(--text-color)"
  />
  <text
    x="21.923828"
    y="89.580078"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
    style="line-height:1.25"
    xml:space="preserve"><tspan x="21.923828" y="89.580078">1</tspan></text
  >
  <text
    x="145.21446"
    y="39.051552"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
    style="line-height:1.25"
    xml:space="preserve"><tspan x="21.92382" y="39.051552">X</tspan></text
  >
  <path
    d="m75 75h50"
    fill="none"
    stroke="var(--text-color)"
    stroke-dasharray="4, 8"
    stroke-dashoffset={offset}
  />
  <rect
    x="135"
    y="45"
    width="60"
    height="60"
    fill="none"
    stroke="var(--text-color)"
  />
  <text
    x="145.21446"
    y="39.051552"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
    style="line-height:1.25"
    xml:space="preserve"><tspan x="145.21446" y="39.051552">W</tspan></text
  >
  <text
    x="90.083527"
    y="84.825775"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
    style="line-height:1.25"
    xml:space="preserve"><tspan x="90.083527" y="84.825775">*</tspan></text
  >
  <path
    id="output"
    d="m205 75h285"
    fill="none"
    stroke={weight >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
    stroke-width={Math.abs(weight)}
    stroke-dasharray="4, 8"
    stroke-dashoffset={offset}
  />
  <text
    id="weight"
    x="151.92383"
    y="89.580078"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
    style="line-height:1.25"
    xml:space="preserve"
    ><tspan x="151.92383" y="89.580078">{weight}</tspan></text
  >
  <g id="button-increase" fill="none" stroke="var(--background-color)">
    <rect
      class="svg-button"
      on:click={increaseWeight}
      x="135"
      y="120"
      width="20"
      height="20"
      fill="var(--main-color-2)"
    />
    <path d="m145 135v-10" marker-end="url(#TriangleOutM)" stroke-width="1px" />
  </g>
  <g id="button-decrease" fill="none" stroke="var(--background-color)">
    <rect
      class="svg-button"
      on:click={decreaseWeight}
      x="175"
      y="120"
      width="20"
      height="20"
      fill="var(--main-color-1)"
    />
    <path d="m185 125v10" marker-end="url(#marker81625)" stroke-width="1px" />
  </g>
</svg>

<style>
  .svg-button {
    cursor: pointer;
  }
</style>
