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
  <g id="chip" transform="translate(125,-300)">
    <rect
      x="198.92"
      y="436.85"
      width="48.154"
      height="48.154"
      ry="8.1539"
      fill="var(--main-color-4)"
      stroke="black"
    />
    <g fill="none">
      <g stroke="black" stroke-width="1px">
        <path d="m238 427v10" marker-start="url(#DotS)" />
        <path d="m228 427v10" marker-start="url(#DotS)" />
        <path d="m218 427v10" marker-start="url(#DotS)" />
        <path d="m208 427v10" marker-start="url(#DotS)" />
        <path d="m238 485v10" marker-end="url(#DotS)" />
        <path d="m228 485v10" marker-end="url(#DotS)" />
        <path d="m218 485v10" marker-end="url(#DotS)" />
        <path d="m208 485v10" marker-end="url(#DotS)" />
        <path d="m258 475h-10" marker-start="url(#DotS)" />
        <path d="m258 465h-10" marker-start="url(#DotS)" />
        <path d="m258 455h-10" marker-start="url(#DotS)" />
        <path d="m258 445h-10" marker-start="url(#DotS)" />
        <path d="m198 475h-10" marker-end="url(#DotS)" />
        <path d="m198 465h-10" marker-end="url(#DotS)" />
        <path d="m198 455h-10" marker-end="url(#DotS)" />
        <path d="m198 445h-10" marker-end="url(#DotS)" />
      </g>
      <g stroke="#000">
        <g>
          <ellipse cx="207.16" cy="443.75" rx="2.6052" ry="2.6028" />
          <ellipse cx="238.03" cy="443.47" rx="2.6052" ry="2.6028" />
          <ellipse cx="207.76" cy="478.26" rx="2.6052" ry="2.6028" />
          <ellipse cx="238.36" cy="478.25" rx="2.6052" ry="2.6028" />
        </g>
        <g stroke-width="1px">
          <path d="m205 455h10" />
          <path d="m205 460h10" />
          <path d="m205 465h10" />
          <path d="m225 470h10" />
          <path d="m230 465h10" />
          <path d="m230 455h10" />
        </g>
        <rect x="215" y="440" width="15" height="5" ry="0" />
      </g>
    </g>
  </g>
  <g id="top-lines" fill="none" stroke="black" stroke-width="1px">
    <path d="m300 60v40" />
    <path d="m310 60v40" />
    <path d="m320 60v40" />
    <path d="m330 60v40" />
    <path d="m340 60v40" />
    <path d="m350 60v40" />
    <path d="m360 60v40" />
    <path d="m370 60v40" />
    <path d="m380 60v40" />
    <path d="m260 230v-60h40" />
    <path d="m450 70v70h-50" />
    <path d="m460 230v-50h-60" />
  </g>
</svg>
