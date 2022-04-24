<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import { onMount } from "svelte";

  onMount(() => {
    let interval = setInterval(() => {
      offset += 40;
    }, 300);

    return () => clearInterval(interval);
  });
  let offset = 0;
</script>

<SvgContainer maxWidth="500px">
  <svg version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
    <g id="agent" fill="none" stroke="var(--text-color)">
      <g>
        <rect x="451.86" y="64.657" width="29.624" height="29.623" ry="0" />
        <rect x="454.82" y="67.619" width="23.698" height="23.698" ry="0" />
        <circle
          cx="466.67"
          cy="79.468"
          r="4.5016"
          fill-rule="evenodd"
          stroke-linejoin="round"
        />
      </g>
      <g>
        <g>
          <path d="m456.71 63.966v-6.1912h-6.1912v-6.1912" />
          <path d="m462.9 63.966v-12.382" />
          <path d="m469.1 63.966v-12.382" />
        </g>
        <path d="m475.29 63.967v-6.1913h6.1912v-6.1913" />
        <g>
          <path d="m481.48 70.158h6.1912v-6.1912h6.1912" />
          <path d="m481.48 76.349h12.382" />
          <path d="m481.48 82.54h12.382" />
          <path d="m481.48 88.731h6.1912v6.1912h6.1912" />
          <path d="m450.52 70.158h-6.1912v-6.1912h-6.1912" />
          <path d="m450.52 76.349h-12.382" />
          <path d="m450.52 82.54h-12.382" />
          <path d="m450.52 88.731h-6.1912v6.1912h-6.1912" />
          <path d="m456.71 94.922v6.1912h-6.1912v6.1912" />
          <path d="m462.9 94.922v12.382" />
          <path d="m469.1 94.922v12.382" />
          <path d="m475.29 94.922v6.1912h6.1912v6.1912" />
        </g>
      </g>
    </g>
    <g id="robot" fill="none" stroke="var(--text-color)">
      <g
        id="legs"
        transform="translate(243.74)"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <rect
          transform="rotate(90)"
          x="189.87"
          y="-68.075"
          width="9.6314"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="rotate(90)"
          x="189.87"
          y="-26.273"
          width="9.6314"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="rotate(90)"
          x="144.97"
          y="-68.075"
          width="41.736"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="rotate(90)"
          x="144.97"
          y="-26.273"
          width="41.736"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="rotate(90)"
          x="101.26"
          y="-68.075"
          width="41.736"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="rotate(90)"
          x="101.26"
          y="-26.273"
          width="41.736"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
      </g>
      <g
        id="body"
        transform="translate(243.74)"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <rect
          x="13.448"
          y="62.557"
          width="57.816"
          height="13.912"
          style="paint-order:markers fill stroke"
        />
        <rect
          x="13.448"
          y="45.977"
          width="57.816"
          height="13.912"
          style="paint-order:markers fill stroke"
        />
        <rect
          x="13.448"
          y="29.396"
          width="57.816"
          height="13.912"
          style="paint-order:markers fill stroke"
        />
        <rect
          x="18.801"
          y="79.137"
          width="47.11"
          height="20.333"
          style="paint-order:markers fill stroke"
        />
      </g>
      <g
        id="arms"
        transform="translate(243.74)"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <rect
          transform="rotate(90)"
          x="30.167"
          y="-10.136"
          width="9.6314"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="matrix(0,1,1,0,0,0)"
          x="30.584"
          y="74.576"
          width="9.6314"
          height="9.6361"
          style="paint-order:markers fill stroke"
        />
        <rect
          x="1.425"
          y="43.824"
          width="7.4947"
          height="27.824"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="scale(-1,1)"
          x="-83.287"
          y="44.241"
          width="7.4947"
          height="27.824"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="matrix(.99997 -.0075742 .0075215 .99997 0 0)"
          x=".85526"
          y="75.756"
          width="7.286"
          height="27.77"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="matrix(-.99999 .0041997 .0042671 .99999 0 0)"
          x="-82.846"
          y="75.723"
          width="7.3762"
          height="27.793"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="rotate(90)"
          x="105.2"
          y="-8.3844"
          width="6.4209"
          height="6.424"
          style="paint-order:markers fill stroke"
        />
        <rect
          transform="rotate(90)"
          x="105.2"
          y="-82.751"
          width="6.4209"
          height="6.424"
          style="paint-order:markers fill stroke"
        />
      </g>
      <g id="head" transform="translate(243.74)">
        <rect
          x="29.508"
          y=".5"
          width="25.696"
          height="25.684"
          ry="0"
          stroke-linecap="round"
          stroke-linejoin="round"
          style="paint-order:markers fill stroke"
        />
        <path d="m42.356 8.3955v10.702" />
      </g>
    </g>
    <path
      id="connection"
      d="m426 75h-90"
      fill="none"
      stroke="var(--text-color)"
      stroke-dasharray="2, 4"
      stroke-dashoffset={offset}
    />
    <g id="batteries" stroke="var(--text-color)" stroke-linejoin="round">
      <rect
        transform="rotate(90)"
        x="171.83"
        y="-115.62"
        width="25"
        height="63.617"
        ry="4.8936"
        fill="none"
        stroke-width="1"
      />
      <g fill="none">
        <rect
          transform="rotate(90)"
          x="174.7"
          y="-104.89"
          width="19.279"
          height="5.4133"
          ry="0"
        />
        <rect
          transform="rotate(90)"
          x="174.7"
          y="-96.067"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="174.7"
          y="-87.241"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="174.7"
          y="-78.414"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="174.7"
          y="-69.587"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="174.7"
          y="-60.761"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="176.83"
          y="-120.62"
          width="15"
          height="5"
        />
        <rect
          transform="rotate(90)"
          x="174.7"
          y="-112.89"
          width="19.279"
          height="5.4133"
          ry="0"
        />
      </g>
      <rect
        transform="rotate(90)"
        x="143.83"
        y="-115.62"
        width="25"
        height="63.617"
        ry="4.8936"
        fill="none"
        stroke-width="1"
      />
      <g fill="none">
        <rect
          transform="rotate(90)"
          x="146.7"
          y="-104.89"
          width="19.279"
          height="5.4133"
          ry="0"
        />
        <rect
          transform="rotate(90)"
          x="146.7"
          y="-96.067"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="146.7"
          y="-87.241"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="146.7"
          y="-78.414"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="146.7"
          y="-69.587"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="146.7"
          y="-60.761"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="148.83"
          y="-120.62"
          width="15"
          height="5"
        />
        <rect
          transform="rotate(90)"
          x="146.7"
          y="-112.89"
          width="19.279"
          height="5.4133"
          ry="0"
        />
      </g>
      <rect
        transform="rotate(90)"
        x="117.83"
        y="-115.62"
        width="25"
        height="63.617"
        ry="4.8936"
        fill="none"
        stroke-width="1"
      />
      <g fill="none">
        <rect
          transform="rotate(90)"
          x="120.7"
          y="-104.89"
          width="19.279"
          height="5.4133"
          ry="0"
        />
        <rect
          transform="rotate(90)"
          x="120.7"
          y="-96.067"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="120.7"
          y="-87.241"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="120.7"
          y="-78.414"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="120.7"
          y="-69.587"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="120.7"
          y="-60.761"
          width="19.279"
          height="5.4133"
        />
        <rect
          transform="rotate(90)"
          x="122.83"
          y="-120.62"
          width="15"
          height="5"
        />
        <rect
          transform="rotate(90)"
          x="120.7"
          y="-112.89"
          width="19.279"
          height="5.4133"
          ry="0"
        />
      </g>
    </g>
    <g id="sparks" fill="none" stroke="var(--text-color)" stroke-width="1px">
      <path d="m85.269 106.99v-13.502" />
      <path d="m96.258 106.66 6.2446-10.545" />
      <path d="m111.01 108.75 10.389-11.87" />
      <path d="m70.604 104.41-7.2062-12.956" />
      <path d="m58.697 105.93-12.772-10.241" />
      <path d="m44.772 119.77-11.367-8.2351" />
      <path d="m45.065 129.08-17.702-2.6742" />
      <path d="m41.728 140.73-16.064 0.5" />
      <path d="m42.142 151.62-14.957 5.7583" />
      <path d="m37.988 167.94-17.506 4.6349" />
      <path d="m44.09 182.83-18.866 3.4485" />
      <path d="m121.14 115.96 13.629-4.1756" />
      <path d="m127.85 129.11 13.056-1" />
      <path d="m131.54 144.72 13.358 0.5" />
      <path d="m134.86 160.38 11.758-3.6032" />
      <path d="m133.74 172.81 13.268 2.4387" />
      <path d="m133.35 183.45 13.832 1.9768" />
      <path d="m135.63 194.65h13.316" />
      <path d="m43.768 193.9-15.104 1.7727" />
    </g>
  </svg>
</SvgContainer>

