<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Button from "$lib/Button.svelte";
  import { draw } from "svelte/transition";

  let showCodeInput = true;
  let showFunctionInput = true;
  let showFunctionOutput = true;
  let showEvaluate = true;
  let showImprove = true;

  let disabled = false;

  function runSimulation() {
    disabled = true;
    // input code
    showCodeInput = false;
    // input to function
    showFunctionInput = false;
    // output from function
    showFunctionOutput = false;
    // output feedback
    showEvaluate = false;
    showImprove = false;

    setTimeout(() => {
      showCodeInput = true;
      showFunctionInput = true;
      showFunctionOutput = true;
      showEvaluate = true;
      showImprove = true;
      disabled = false;
    }, 1000);
  }
</script>

<SvgContainer maxWidth={"500px"}>
  <svg version="1.1" viewBox="0 0 350 290" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="TriangleOutM" overflow="visible" orient="auto">
        <path
          transform="scale(.4)"
          d="m5.77 0-8.65 5v-10l8.65 5z"
          fill="context-stroke"
          fill-rule="evenodd"
          stroke="context-stroke"
          stroke-width="1pt"
        />
      </marker>
      <marker id="DotM" overflow="visible" orient="auto">
        <path
          transform="matrix(.4 0 0 .4 2.96 .4)"
          d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
          fill="context-stroke"
          fill-rule="evenodd"
          stroke="context-stroke"
          stroke-width="1pt"
        />
      </marker>
      <marker id="DotS" overflow="visible" orient="auto">
        <path
          transform="matrix(.2 0 0 .2 1.48 .2)"
          d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
          fill="context-stroke"
          fill-rule="evenodd"
          stroke="context-stroke"
          stroke-width="1pt"
        />
      </marker>
    </defs>
    <g>
      <g id="chip" transform="translate(-6,-34)" stroke="#000">
        <rect
          x="118.92"
          y="211.85"
          width="48.154"
          height="48.154"
          ry="8.1539"
          fill="var(--main-color-1)"
        />
        <g fill="none">
          <g stroke-width="1px">
            <path d="m158 202v10" marker-start="url(#DotS)" />
            <path d="m148 202v10" marker-start="url(#DotS)" />
            <path d="m138 202v10" marker-start="url(#DotS)" />
            <path d="m128 202v10" marker-start="url(#DotS)" />
            <path d="m158 260v10" marker-end="url(#DotS)" />
            <path d="m148 260v10" marker-end="url(#DotS)" />
            <path d="m138 260v10" marker-end="url(#DotS)" />
            <path d="m128 260v10" marker-end="url(#DotS)" />
            <path d="m178 250h-10" marker-start="url(#DotS)" />
            <path d="m178 240h-10" marker-start="url(#DotS)" />
            <path d="m178 230h-10" marker-start="url(#DotS)" />
            <path d="m178 220h-10" marker-start="url(#DotS)" />
            <path d="m118 250h-10" marker-end="url(#DotS)" />
            <path d="m118 240h-10" marker-end="url(#DotS)" />
            <path d="m118 230h-10" marker-end="url(#DotS)" />
            <path d="m118 220h-10" marker-end="url(#DotS)" />
          </g>
          <g>
            <ellipse cx="127.16" cy="218.75" rx="2.6052" ry="2.6028" />
            <ellipse cx="158.03" cy="218.47" rx="2.6052" ry="2.6028" />
            <ellipse cx="127.76" cy="253.26" rx="2.6052" ry="2.6028" />
            <ellipse cx="158.36" cy="253.25" rx="2.6052" ry="2.6028" />
          </g>
          <g stroke-width="1px">
            <path d="m125 230h10" />
            <path d="m125 235h10" />
            <path d="m125 240h10" />
            <path d="m145 245h10" />
            <path d="m150 240h10" />
            <path d="m150 230h10" />
          </g>
          <rect x="135" y="215" width="15" height="5" ry="0" />
        </g>
      </g>
      {#if showFunctionInput}
        <g
          id="function-inputs"
          transform="translate(0,-26)"
          fill="none"
          stroke="#000"
        >
          <path
            in:draw={{ duration: 500, delay: 1000 }}
            d="m8 224h70"
            marker-start="url(#DotM)"
          />
          <path
            in:draw={{ duration: 500, delay: 1000 }}
            d="m8 236h70"
            marker-start="url(#DotM)"
          />
          <path
            in:draw={{ duration: 500, delay: 1000 }}
            d="m137 302v-50"
            fill="none"
            marker-start="url(#DotM)"
            stroke="#000"
            stroke-width="1px"
          />
        </g>
      {/if}
      {#if showCodeInput}
        <path
          id="code-input"
          in:draw={{ duration: 500 }}
          d="m138 116v44"
          fill="none"
          marker-end="url(#TriangleOutM)"
          stroke="#000"
        />
      {/if}
      <g id="code" transform="translate(0,18)" stroke="#000">
        <rect
          x="108"
          y="4"
          width="60"
          height="90"
          fill="var(--main-color-2)"
          stroke-width="2"
        />
        <g fill="none" stroke-width="1px">
          <path d="m118 34h40" />
          <path d="m118 44h40" />
          <path d="m118 54h40" />
          <path d="m118 64h40" />
          <path d="m118 74h40" />
          <path d="m118 84h40" />
          <path d="m123 9-5 5 5 5" />
          <path d="m131 9-5 10" />
          <path d="m132 9 5 5-5 5" />
        </g>
      </g>
      {#if showFunctionOutput}
        <path
          in:draw={{ duration: 500, delay: 2000 }}
          d="m198 200h122"
          fill="none"
          marker-end="url(#marker2202)"
          stroke="#000"
        />
      {/if}
      {#if showEvaluate}
        <path
          in:draw={{ duration: 500, delay: 3000 }}
          d="m45 151.67v-141.67h220v165"
          fill="none"
          marker-end="url(#DotM)"
          marker-start="url(#DotM)"
          stroke="#000"
          stroke-width="1px"
        />
      {/if}
      {#if showImprove}
        <path
          in:draw={{ duration: 500, delay: 4000 }}
          d="m320 210v75h-175"
          fill="none"
          marker-end="url(#TriangleOutM)"
          stroke="#000"
          stroke-width="1px"
        />
      {/if}
      <g fill="#000000" font-size="16.673px">
        <text
          x="12.546965"
          y="236.48779"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="12.546965" y="236.48779">2. Inputs</tspan></text
        >
        <text
          x="242.54794"
          y="194.23212"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="242.54794" y="194.23212">3. Outputs</tspan></text
        >
        <text
          x="248.58961"
          y="258.23315"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="248.58961" y="258.23315" /><tspan x="230" y="279.0744"
            >5. Improve</tspan
          ></text
        >
        <text
          x="181.68901"
          y="42.35067"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="181.68901" y="42.35067">1. Code</tspan></text
        >
      </g>
      <text
        x="72.882713"
        y="287.15833"
        fill="#000000"
        font-size="16.673px"
        style="line-height:1.25"
        xml:space="preserve"
        ><tspan x="60" y="287.15833">2. Weights</tspan></text
      >
      <text
        x="10.118352"
        y="169.74382"
        fill="#000000"
        font-size="16.673px"
        style="line-height:1.25"
        xml:space="preserve"
        ><tspan x="10.118352" y="169.74382">2. Expected </tspan><tspan
          x="10.118352"
          y="190.58507">Outputs</tspan
        ></text
      >
      <text
        x="269.32263"
        y="101.85865"
        fill="#000000"
        font-size="16.673px"
        style="line-height:1.25"
        xml:space="preserve"
        ><tspan x="269.32263" y="101.85865">4. Evaluate</tspan></text
      >
    </g>
  </svg>
</SvgContainer>
<Button on:click={runSimulation} value="Simulate Process" />

<style>
  text {
    font-size: 15px;
  }
</style>
