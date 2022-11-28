<script>
  import { scaleLinear } from "d3-scale";
  let width = 400;
  let height = 200;

  let state = 0;
  let action = 0;

  const allPoints = [
    [
      [
        { state: 0, probability: 1 },
        { state: 1, probability: 0 },
        { state: 2, probability: 0 },
        { state: 3, probability: 0 },
      ],
      [
        { state: 0, probability: 0.2 },
        { state: 1, probability: 0.8 },
        { state: 2, probability: 0.0 },
        { state: 3, probability: 0.0 },
      ],
    ],
    [
      [
        { state: 0, probability: 0.6 },
        { state: 1, probability: 0.3 },
        { state: 2, probability: 0.1 },
        { state: 3, probability: 0 },
      ],
      [
        { state: 0, probability: 0 },
        { state: 1, probability: 0.4 },
        { state: 2, probability: 0.6 },
        { state: 3, probability: 0.0 },
      ],
    ],
    [
      [
        { state: 0, probability: 0 },
        { state: 1, probability: 0.8 },
        { state: 2, probability: 0.2 },
        { state: 3, probability: 0 },
      ],
      [
        { state: 0, probability: 0.0 },
        { state: 1, probability: 0.0 },
        { state: 2, probability: 0.4 },
        { state: 3, probability: 0.6 },
      ],
    ],
    [
      [
        { state: 0, probability: 0 },
        { state: 1, probability: 0 },
        { state: 2, probability: 0 },
        { state: 3, probability: 1 },
      ],
      [
        { state: 0, probability: 0 },
        { state: 1, probability: 0 },
        { state: 2, probability: 0 },
        { state: 3, probability: 1 },
      ],
    ],
  ];
  $: points = allPoints[state][action];
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const padding = { top: 20, right: 15, bottom: 20, left: 25 };

  $: xScale = scaleLinear()
    .domain([0, 4])
    .range([padding.left, width - padding.right]);

  $: yScale = scaleLinear()
    .domain([0, Math.max.apply(null, yTicks)])
    .range([height - padding.bottom, padding.top]);

  $: innerWidth = width - (padding.left + padding.right);
  $: barWidth = innerWidth / 8;
</script>

<div class="container">
  <svg viewBox="0 0 {width} {height}">
    <!-- y axis -->
    <g class="axis y-axis">
      {#each yTicks as tick}
        <g class="tick tick-{tick}" transform="translate(0, {yScale(tick)})">
          <line stroke="var(--text-color)" x2="100%" />
          <text fill="var(--text-color)" y="-4">{tick}</text>
        </g>
      {/each}
    </g>

    <!-- x axis -->
    <g class="axis x-axis">
      {#each points as point, i}
        <g class="tick" transform="translate({xScale(i)},{height})">
          <text fill="var(--text-color)" x={barWidth / 2} y="-4"
            >{point.state}</text
          >
        </g>
      {/each}
    </g>

    <g class="bars">
      {#each points as point, i}
        <rect
          x={xScale(i) + 8}
          y={yScale(point.probability)}
          width={barWidth - 8}
          height={yScale(0) - yScale(point.probability)}
          fill="var(--main-color-1)"
        />
      {/each}
    </g>
  </svg>
</div>
<form>
  <div class="flex-space">
    <div>
      <label for="state">State:</label>
      <select bind:value={state} id="state">
        <option value={0}>0</option>
        <option value={1}>1</option>
        <option value={2}>2</option>
        <option value={3}>3</option>
      </select>
    </div>
    <div>
      <label for="action">Action:</label>
      <select bind:value={action} id="action">
        <option value={0}>0</option>
        <option value={1}>1</option>
      </select>
    </div>
  </div>
</form>

<style>
  label {
    text-transform: uppercase;
    font-size: 20px;
    margin-right: 15px;
  }

  select {
    padding: 2px;
    border: 1px solid var(--main-color-1);
    outline: none;
    text-align: center;
    cursor: pointer;
    font-size: 20px;
    width: 50px;
    background-color: var(--background-color);
    color: var(--text-color);
  }

  .container {
    width: 100%;
    max-width: 600px;
    display: flex;
    justify-self: center;
  }
  text {
    font-size: 15px;
  }
  line {
    stroke-width: 0.2px;
    opacity: 0.5;
  }
</style>
