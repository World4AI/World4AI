<script>
  import { onMount } from "svelte";
  // offset for stroke-dashoffset to emulate movement
  let offset = 0;

  onMount(() => {
    let interval = setInterval(() => {
      offset += 20;
    }, 300);
    return () => {
      clearInterval(interval);
    };
  });

  let weight1 = 1;
  let weight2 = 1;
  function increaseWeight1() {
    if (weight1 < 9) {
      weight1 += 1;
    }
  }
  function decreaseWeight1() {
    if (weight1 > -9) {
      weight1 -= 1;
    }
  }
  function increaseWeight2() {
    if (weight2 < 9) {
      weight2 += 1;
    }
  }
  function decreaseWeight2() {
    if (weight2 > -9) {
      weight2 -= 1;
    }
  }
</script>

<svg version="1.1" viewBox="0 0 500 250" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="TriangleOutM" overflow="visible" orient="auto">
      <path
        transform="scale(.4)"
        d="m5.77 0-8.65 5v-10z"
        fill="context-stroke"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
    <marker id="marker81625" overflow="visible" orient="auto">
      <path
        transform="scale(.4)"
        d="m5.77 0-8.65 5v-10z"
        fill="context-stroke"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g fill="none" stroke="var(--text-color)">
    <rect x="14" y="4.8718" width="60" height="60" />
    <rect x="14" y="184.87" width="60" height="60" />
    <rect x="144" y="4.8718" width="60" height="60" />
    <rect x="144" y="184.87" width="60" height="60" />
  </g>
  <g
    transform="matrix(.54961 0 0 .54961 80.846 51.367)"
    fill="var(--text-color)"
    font-family="sans-serif"
  >
    <text
      x="145.21484"
      y="61.45192"
      font-size="40px"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="145.21484" y="61.45192">W</tspan></text
    >
    <text
      x="177.52051"
      y="71.777275"
      font-size="27.606px"
      stroke-width=".69015"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="177.52051" y="71.777275" stroke-width=".69015">1</tspan></text
    >
  </g>
  <g
    transform="matrix(.54961 0 0 .54961 80.879 55.206)"
    fill="var(--text-color)"
    font-family="sans-serif"
  >
    <text
      x="145.21484"
      y="221.45192"
      font-size="40px"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="145.21484" y="221.45192">W</tspan></text
    >
    <text
      x="177.52051"
      y="233.83195"
      font-size="27.606px"
      stroke-width=".69015"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="177.52051" y="233.83195" stroke-width=".69015">2</tspan></text
    >
  </g>
  <g fill="var(--text-color)" font-family="sans-serif" font-size="40px">
    <text
      x="99.083527"
      y="43.534355"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="99.083527" y="43.534355">*</tspan></text
    >
    <text
      x="99.083527"
      y="249.55052"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="99.083527" y="249.55052">*</tspan></text
    >
    <text
      x="247.59177"
      y="139.4109"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="253" y="139.4109">+</tspan></text
    >
  </g>
  <g id="signals" fill="none" stroke="var(--text-color)">
    <g stroke-dasharray="4, 8">
      <path d="m84 34.872h50" stroke-dashoffset={offset} />
      <path d="m84 214.87h50" stroke-dashoffset={offset} />
      <path
        d="m209 35 55-0.12816v75.128"
        stroke-dashoffset={offset}
        stroke-width={Math.abs(weight1)}
        stroke={weight1 >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
      />
      <path
        d="m209 215 55-0.12816v-69.872"
        stroke-dashoffset={offset}
        stroke-width={Math.abs(weight2)}
        stroke={weight2 >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
      />
      <path
        id="added_signal"
        stroke-width={Math.abs(weight1 + weight2)}
        stroke={weight1 + weight2 >= 0
          ? "var(--main-color-2)"
          : "var(--main-color-1)"}
        d="m289 126.87h206"
        stroke-dashoffset={offset}
      />
    </g>
  </g>
  <g
    transform="matrix(.54961 0 0 .54961 -44.842 49.707)"
    fill="var(--text-color)"
    font-family="sans-serif"
  >
    <text
      x="145.21484"
      y="61.45192"
      font-size="40px"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="145.21484" y="61.45192">X</tspan></text
    >
    <text
      x="171.67813"
      y="71.777275"
      font-size="27.606px"
      stroke-width=".69015"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="171.67813" y="71.777275" stroke-width=".69015">1</tspan></text
    >
  </g>
  <g
    transform="matrix(.54961 0 0 .54961 -51.97 40.464)"
    fill="var(--text-color)"
    font-family="sans-serif"
  >
    <text
      x="156.09352"
      y="248.90216"
      font-size="40px"
      style="line-height:1.25"
      xml:space="preserve"><tspan x="156.09352" y="248.90216">X</tspan></text
    >
    <text
      x="180.38985"
      y="261.2822"
      font-size="27.606px"
      stroke-width=".69015"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="180.38985" y="261.2822" stroke-width=".69015">2</tspan></text
    >
  </g>
  <g id="buttons" fill="none" stroke="var(--background-color)">
    <rect
      class="svg-button"
      x="153"
      y="96"
      width="20"
      height="20"
      on:click={increaseWeight1}
      fill="var(--main-color-2)"
    />
    <path d="m163 111v-10" marker-end="url(#TriangleOutM)" stroke-width="1px" />
    <rect
      class="svg-button"
      x="179"
      y="96"
      width="20"
      height="20"
      on:click={decreaseWeight1}
      fill="var(--main-color-1)"
    />
    <path d="m189 101v10" marker-end="url(#marker81625)" stroke-width="1px" />
    <rect
      class="svg-button"
      x="153"
      y="135"
      width="20"
      height="20"
      fill="var(--main-color-2)"
      on:click={increaseWeight2}
    />
    <path d="m163 150v-10" marker-end="url(#TriangleOutM)" stroke-width="1px" />
    <rect
      class="svg-button"
      x="179"
      y="135"
      width="20"
      height="20"
      on:click={decreaseWeight2}
      fill="var(--main-color-1)"
    />
    <path d="m189 140v10" marker-end="url(#marker81625)" stroke-width="1px" />
  </g>
  <text
    x="30.923828"
    y="49.45192"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
    style="line-height:1.25"
    xml:space="preserve"><tspan x="30.923828" y="49.45192">1</tspan></text
  >
  <text
    x="30.923828"
    y="229.45192"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
    style="line-height:1.25"
    xml:space="preserve"><tspan x="30.923828" y="229.45192">1</tspan></text
  >
  <g
    id="weights"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="40px"
  >
    <text
      x="160.92383"
      y="49.45192"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="160.92383" y="49.45192">{weight1}</tspan></text
    >
    <text
      x="160.92383"
      y="229.45192"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="160.92383" y="229.45192">{weight2}</tspan></text
    >
  </g>
</svg>

<style>
  .svg-button {
    cursor: pointer;
  }
</style>
