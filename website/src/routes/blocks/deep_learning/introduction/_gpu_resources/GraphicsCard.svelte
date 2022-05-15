<script>
  import { onMount } from "svelte";
  let angle = 0;

  onMount(() => {
    const interval = setInterval(() => {
      angle += 1;
    }, 30);

    return () => {
      clearInterval(interval);
    };
  });
</script>

<svg version="1.1" viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="DotS" overflow="visible" orient="auto">
      <path
        transform="matrix(.2 0 0 .2 1.48 .2)"
        d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
        fill="black"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g id="parts">
    <path
      d="m13 10h30v280"
      fill="none"
      stroke="black"
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="5"
    />
    <g fill="var(--main-color-4)" stroke="black">
      <rect x="18" y="133" width="20" height="90" />
      <rect x="17" y="112" width="20" height="10" />
      <rect x="17" y="232" width="20" height="10" />
      <rect x="70" y="60" width="420" height="180" />
      <rect x="48" y="70" width="20" height="180" />
      <rect x="100" y="242" width="270" height="20" />
    </g>
  </g>
  <g id="lines" stroke="black" stroke-width="1px">
    <path d="m135 262v-20" fill="#fff" />
    <path d="m140 262v-20" fill="#fff" />
    <path d="m145 262v-20" fill="none" />
    <path d="m165 262v-20" fill="#fff" />
    <path d="m170 262v-20" fill="#fff" />
    <path d="m175 262v-20" fill="none" />
    <path d="m180 262v-20" fill="none" />
    <path d="m190 262v-20" fill="#fff" />
    <path d="m195 262v-20" fill="#fff" />
    <path d="m200 262v-20" fill="none" />
    <path d="m205 262v-20" fill="none" />
    <path d="m215 262v-20" fill="#fff" />
    <path d="m220 262v-20" fill="#fff" />
    <path d="m225 262v-20" fill="none" />
    <path d="m230 262v-20" fill="none" />
    <path d="m240 262v-20" fill="#fff" />
    <path d="m245 262v-20" fill="#fff" />
    <path d="m250 262v-20" fill="none" />
    <path d="m255 262v-20" fill="none" />
    <path d="m285 262v-20" fill="#fff" />
    <path d="m290 262v-20" fill="#fff" />
    <path d="m295 262v-20" fill="none" />
    <path d="m300 262v-20" fill="none" />
    <path d="m325 262v-20" fill="#fff" />
    <path d="m330 262v-20" fill="#fff" />
    <path d="m335 262v-20" fill="none" />
    <path d="m340 262v-20" fill="none" />
    <path d="m350 262v-20" fill="#fff" />
    <path d="m355 262v-20" fill="#fff" />
    <path d="m360 262v-20" fill="none" />
    <path d="m365 262v-20" fill="none" />
  </g>
  <g id="fan" transform="rotate({angle} 167.38 149.18)">
    <circle
      cx="167.38"
      cy="149.18"
      r="63.738"
      fill="var(--main-color-4)"
      stroke="black"
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="3"
    />
    <g id="rotors" fill="yellow" stroke="black" stroke-width="2px">
      <g transform="rotate(1.2088 -7667.5 -23237)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(-43.791 625.3 675.55)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(-88.791 500.26 314.98)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(226.21 449.56 168.8)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(181.21 414.09 66.521)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(136.21 378.93 -34.866)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(91.209 329.73 -176.75)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(46.209 213.38 -512.25)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
    </g>
    <circle cx="167.38" cy="149.18" r="6.7554" fill="black" />
  </g>
  <g id="fan" transform="rotate({angle * 2.5} 387.38 149.18) translate(220 0)">
    <circle
      cx="167.38"
      cy="149.18"
      r="63.738"
      fill="var(--main-color-4)"
      stroke="black"
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="3"
    />
    <g id="rotors" fill="yellow" stroke="black" stroke-width="2px">
      <g transform="rotate(1.2088 -7667.5 -23237)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(-43.791 625.3 675.55)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(-88.791 500.26 314.98)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(226.21 449.56 168.8)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(181.21 414.09 66.521)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(136.21 378.93 -34.866)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(91.209 329.73 -176.75)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
      <g transform="rotate(46.209 213.38 -512.25)">
        <path d="m658.93-18.626v-53.374" />
        <path d="m657.95-18.626 27.054-41.374" />
        <path d="m659-72 26 12" stroke-linecap="square" />
      </g>
    </g>
    <circle cx="167.38" cy="149.18" r="6.7554" fill="black" />
  </g>
</svg>
