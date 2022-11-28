<script>
  import { onMount } from "svelte";
  import { draw } from "svelte/transition";

  let forward = true;
  let backward = false;

  onMount(() => {
    const interval = setInterval(() => {
      forward = !forward;
      backward = !backward;
    }, 4000);

    return () => {
      clearInterval(interval);
    };
  });
</script>

<svg version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
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
  <g stroke="#000">
    <g
      id="neuron-left-top"
      transform="translate(-35)"
      fill="var(--main-color-3)"
    >
      <rect x="186.67" y="16.667" width="46.667" height="46.667" ry="5.8333" />
      <g id="neuron-lines" stroke-width="1px">
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
      <g id="neuron-circles">
        <circle cx="192.5" cy="22.5" r="2.4442" />
        <circle cx="227.51" cy="22.5" r="2.4442" />
        <circle cx="192.5" cy="57.398" r="2.4442" />
        <circle cx="227.51" cy="57.398" r="2.4442" />
      </g>
      <g stroke-width=".5">
        <path d="m210 34.167h17.5" />
        <path d="m210 40h17.5" />
        <path d="m210 45.833h17.5" />
        <path d="m192.5 51.667h11.667" />
        <path d="m192.5 48.75h11.667" />
        <path d="m192.5 45.833h11.667" />
      </g>
      <rect x="210" y="51.667" width="11.667" height="5.8333" />
      <g id="large-chip" stroke-width=".5">
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
        <rect x="190" y="31.667" width="5" height="5" stroke-width=".33333" />
        <g stroke-width=".33333px">
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
        <rect x="200" y="36.667" width="5" height="5" stroke-width=".33333" />
        <g stroke-width=".33333px">
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
    <g
      id="neuron-left-bot"
      transform="translate(-35 117.9)"
      fill="var(--main-color-3)"
    >
      <rect x="186.67" y="16.667" width="46.667" height="46.667" ry="5.8333" />
      <g stroke-width="1px">
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
      <g>
        <circle cx="192.5" cy="22.5" r="2.4442" />
        <circle cx="227.51" cy="22.5" r="2.4442" />
        <circle cx="192.5" cy="57.398" r="2.4442" />
        <circle cx="227.51" cy="57.398" r="2.4442" />
      </g>
      <g stroke-width=".5">
        <path d="m210 34.167h17.5" />
        <path d="m210 40h17.5" />
        <path d="m210 45.833h17.5" />
        <path d="m192.5 51.667h11.667" />
        <path d="m192.5 48.75h11.667" />
        <path d="m192.5 45.833h11.667" />
      </g>
      <rect x="210" y="51.667" width="11.667" height="5.8333" />
      <g stroke-width=".5">
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
      <rect x="190" y="31.667" width="5" height="5" stroke-width=".33333" />
      <g stroke-width=".33333px">
        <path d="m191.67 30v1.6667" />
        <path d="m193.33 30v1.6667" />
        <path d="m188.33 33.333h1.6667" />
        <path d="m188.33 35h1.6667" />
        <path d="m191.67 38.333v-1.6667" />
        <path d="m193.33 38.333v-1.6667" />
        <path d="m196.67 33.333h-1.6667" />
        <path d="m196.67 35h-1.6667" />
      </g>
      <rect x="200" y="36.667" width="5" height="5" stroke-width=".33333" />
      <g stroke-width=".33333px">
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
    <g
      id="neuron-right"
      transform="translate(128.55 60)"
      fill="var(--main-color-3)"
    >
      <rect x="186.67" y="16.667" width="46.667" height="46.667" ry="5.8333" />
      <g stroke-width="1px">
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
      <g>
        <circle cx="192.5" cy="22.5" r="2.4442" />
        <circle cx="227.51" cy="22.5" r="2.4442" />
        <circle cx="192.5" cy="57.398" r="2.4442" />
        <circle cx="227.51" cy="57.398" r="2.4442" />
      </g>
      <g stroke-width=".5">
        <path d="m210 34.167h17.5" />
        <path d="m210 40h17.5" />
        <path d="m210 45.833h17.5" />
        <path d="m192.5 51.667h11.667" />
        <path d="m192.5 48.75h11.667" />
        <path d="m192.5 45.833h11.667" />
      </g>
      <rect x="210" y="51.667" width="11.667" height="5.8333" />
      <g stroke-width=".5">
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
      <rect x="190" y="31.667" width="5" height="5" stroke-width=".33333" />
      <g stroke-width=".33333px">
        <path d="m191.67 30v1.6667" />
        <path d="m193.33 30v1.6667" />
        <path d="m188.33 33.333h1.6667" />
        <path d="m188.33 35h1.6667" />
        <path d="m191.67 38.333v-1.6667" />
        <path d="m193.33 38.333v-1.6667" />
        <path d="m196.67 33.333h-1.6667" />
        <path d="m196.67 35h-1.6667" />
      </g>
      <rect x="200" y="36.667" width="5" height="5" stroke-width=".33333" />
      <g stroke-width=".33333px">
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
    {#if forward}
      <g id="forward-first" fill="none">
        <path in:draw={{ duration: 1000 }} d="m10 22.5h125" />
        <path in:draw={{ duration: 1000 }} d="m10 57.5h125" />
        <path in:draw={{ duration: 1000 }} d="m10 34.167h124.19" />
        <path in:draw={{ duration: 1000 }} d="m10 45.833h124.19" />
        <path in:draw={{ duration: 1000 }} d="m10 140.4h125" />
        <path in:draw={{ duration: 1000 }} d="m10 175.4h125" />
        <path in:draw={{ duration: 1000 }} d="m10 152.07h124.19" />
        <path in:draw={{ duration: 1000 }} d="m10 163.73h124.19" />
      </g>
      <g id="forward-second" fill="none">
        <path
          in:draw={{ duration: 1000, delay: 1000 }}
          d="m218.91 39.647 76.852 53.743"
        />
        <path
          in:draw={{ duration: 1000, delay: 1000 }}
          d="m218.91 158.76 76.852-53.743"
        />
      </g>
      <path
        in:draw={{ duration: 1000, delay: 2000 }}
        id="forward-third"
        d="m380 100.54h60"
        fill="none"
      />
    {/if}
    {#if backward}
      <path
        id="backward-first"
        stroke="var(--main-color-1)"
        in:draw={{ duration: 1000, delay: 0 }}
        d="m440 100.54h-60"
        fill="none"
      />

      <g id="backward-second" fill="none">
        <path
          stroke="var(--main-color-1)"
          in:draw={{ duration: 1000, delay: 1000 }}
          d="m295.76 93.39-76.852-53.743"
        />
        <path
          d="m295.76 105.01-76.852 53.743"
          stroke="var(--main-color-1)"
          in:draw={{ duration: 1000, delay: 1000 }}
        />
      </g>
    {/if}
  </g>
</svg>
