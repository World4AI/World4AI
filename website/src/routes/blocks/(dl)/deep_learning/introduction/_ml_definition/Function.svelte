<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import { onMount } from "svelte";

  let offset = 0;
  onMount(() => {
    const interval = setInterval(() => {
      offset -= 2;
    }, 200);

    return () => {
      clearInterval(interval);
    };
  });
</script>

<SvgContainer maxWidth="500px">
  <svg version="1.1" viewBox="0 0 800 330" xmlns="http://www.w3.org/2000/svg">
    <defs>
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
    <text
      x="7.7187953"
      y="212.79494"
      fill="#000000"
      font-family="sans-serif"
      font-size="40px"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="7.7187953" y="212.79494">Inputs</tspan></text
    >
    <text
      x="632.1347"
      y="213.05861"
      fill="#000000"
      font-family="sans-serif"
      font-size="40px"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="632.1347" y="213.05861">Outputs</tspan></text
    >
    <g>
      <text
        x="314.66797"
        y="52.91214"
        fill="#000000"
        font-family="sans-serif"
        font-size="40px"
        style="line-height:1.25"
        xml:space="preserve"><tspan x="323" y="52">Function</tspan></text
      >
    </g>
    <g
      id="neuron"
      transform="matrix(3.2411 0 0 3.2411 -280.62 72.732)"
      fill="var(--main-color-3)"
      stroke="#000"
      stroke-width="1.2645"
    >
      <rect
        x="186.67"
        y="16.667"
        width="46.667"
        height="46.667"
        ry="5.8333"
        stroke-width="1.2645"
      />
      <g id="neuron-lines" stroke-width="1.2645px">
        <path d="m192.5 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m204.17 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m215.83 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m227.5 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m233.33 22.5h11.667" marker-end="url(#DotS)" />
        <path d="m233.33 34.167h11.667" marker-end="url(#DotS)" />
        <path d="m233.33 45.833h11.667" marker-end="url(#DotS)" />
        <path d="m233.33 57.5h11.667" marker-end="url(#DotS)" />
        <path d="m227.5 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m215.83 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m204.17 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m192.5 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m186.67 57.5h-11.667" marker-end="url(#DotS)" />
        <path d="m186.67 45.833h-11.667" marker-end="url(#DotS)" />
        <path d="m186.67 34.167h-11.667" marker-end="url(#DotS)" />
        <path d="m186.67 22.5h-11.667" marker-end="url(#DotS)" />
      </g>
      <g id="neuron-circles" stroke-width="1.2645">
        <circle cx="192.5" cy="22.5" r="2.4442" />
        <circle cx="227.51" cy="22.5" r="2.4442" />
        <circle cx="192.5" cy="57.398" r="2.4442" />
        <circle cx="227.51" cy="57.398" r="2.4442" />
      </g>
      <g stroke-width=".63223">
        <path d="m210 34.167h17.5" />
        <path d="m210 40h17.5" />
        <path d="m210 45.833h17.5" />
        <path d="m192.5 51.667h11.667" />
        <path d="m192.5 48.75h11.667" />
        <path d="m192.5 45.833h11.667" />
      </g>
      <rect
        x="210"
        y="51.667"
        width="11.667"
        height="5.8333"
        stroke-width="1.2645"
      />
      <g id="large-chip" stroke-width=".63223">
        <rect x="207.4" y="19.985" width="13.201" height="5.0296" />
        <g>
          <path d="m209 17.5v2.5" />
          <path d="m211.5 17.5v2.5" />
          <path d="m214 17.5v2.5" />
          <path d="m216.5 17.5v2.5" />
          <path d="m219 17.5v2.5" />
          <path d="m219 25v2.5" />
          <path d="m216.5 25v2.5" />
          <path d="m214 25v2.5" />
          <path d="m211.5 25v2.5" />
          <path d="m209 25v2.5" />
        </g>
      </g>
      <g id="small-chip-left">
        <rect x="190" y="31.667" width="5" height="5" stroke-width=".42149" />
        <g stroke-width=".42149px">
          <path d="m191.67 30v1.6667" />
          <path d="m193.33 30v1.6667" />
          <path d="m188.33 33.333h1.6667" />
          <path d="m188.33 35h1.6667" />
          <path d="m191.67 38.333v-1.6667" />
          <path d="m193.33 38.333v-1.6667" />
          <path d="m196.67 33.333h-1.6667" />
          <path d="m196.67 35h-1.6667" />
        </g>
      </g>
      <g id="small-chip-right">
        <rect x="200" y="36.667" width="5" height="5" stroke-width=".42149" />
        <g stroke-width=".42149px">
          <path d="m201.67 35v1.6667" />
          <path d="m203.33 35v1.6667" />
          <path d="m198.33 38.333h1.6667" />
          <path d="m198.33 40h1.6667" />
          <path d="m201.67 43.333v-1.6667" />
          <path d="m203.33 43.333v-1.6667" />
          <path d="m206.67 38.333h-1.6667" />
          <path d="m206.67 40h-1.6667" />
        </g>
      </g>
    </g>
    <path
      d="m140 198h140"
      fill="none"
      stroke="#000"
      stroke-dasharray="15, 15"
      stroke-dashoffset={offset}
      stroke-width="3"
    />
    <path
      d="m520 198h100"
      fill="none"
      stroke="#000"
      stroke-dasharray="15, 15"
      stroke-dashoffset={offset}
      stroke-width="3"
    />
  </svg>
</SvgContainer>
