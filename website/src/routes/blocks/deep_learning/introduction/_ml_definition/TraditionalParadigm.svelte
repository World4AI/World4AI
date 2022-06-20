<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
</script>

<div class="flex-container">
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
        <marker id="marker2202" overflow="visible" orient="auto">
          <path
            transform="matrix(.8 0 0 .8 5.92 .8)"
            d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
            fill="context-stroke"
            fill-rule="evenodd"
            stroke="context-stroke"
            stroke-width="1pt"
          />
        </marker>
      </defs>
      <g stroke="#000">
        <g id="chip">
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
        <g id="function-inputs" fill="none">
          <path d="m8 214h70" marker-start="url(#DotM)" />
          <path d="m8 234h70" marker-start="url(#DotM)" />
          <path d="m8 254h70" marker-start="url(#DotM)" />
        </g>
        <path
          id="code-input"
          d="m138 104v71"
          fill="none"
          marker-end="url(#TriangleOutM)"
        />
        <g id="code">
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
        <path
          id="output"
          d="m201 234h122"
          fill="none"
          marker-end="url(#marker2202)"
        />
        <path
          id="feedback"
          d="m323 225v-211h-145"
          fill="none"
          marker-end="url(#TriangleOutM)"
        />
      </g>
      <g fill="#000000">
        <text
          x="10.456777"
          y="281.07825"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="10.456777" y="281.07825">2. Inputs</tspan></text
        >
        <text
          x="220.04919"
          y="281.07825"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="220.04919" y="281.07825">3. Outputs</tspan></text
        >
        <text
          x="220.04919"
          y="101.23457"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="220.04919" y="101.23457">4. Evaluate</tspan><tspan
            x="220.04919"
            y="126.4732"
            >and
          </tspan><tspan x="220.04919" y="151.71182">Improve</tspan></text
        >
        <text
          x="40.55479"
          y="28.234875"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="40.55479" y="28.234875">1. Code</tspan></text
        >
      </g>
    </svg>
  </SvgContainer>
</div>

<style>
  text {
    font-size: 15px;
  }
</style>
