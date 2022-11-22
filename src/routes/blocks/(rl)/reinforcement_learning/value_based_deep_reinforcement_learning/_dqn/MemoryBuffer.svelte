<script>
  import Button from "$lib/Button.svelte";
  export let prioritized = false;
  let width = 700;
  let height = 250;

  let timeStep = 5;
  let bufferMaxLen = 15;
  let batchLen = 3;
  let priorityLen = 3;

  let maxPriority = 0;
  let priority = [0, 0, 0, 0, 0, 0];
  if (prioritized) {
    maxPriority = 20;
    priority = [20, 16, 16, 10, 8, 8];
  }

  let buffer = [5, 4, 3, 2, 1, 0];
  let batch = [2, 1, 4];

  let boxSize = 35;
  let gap = 5;
  let yGap = 40;
  let containerGap = 10;
  let bufferLeftStart = (width - (boxSize + gap) * bufferMaxLen) / 2;
  let batchLeftStart = (width - (boxSize + gap) * batchLen) / 2;
  let y = height - boxSize - yGap;
  let containerY = height - boxSize - priorityLen * maxPriority - yGap;
  let yBatch = boxSize;

  function handleClick() {
    timeStep += 1;
    if (buffer.length >= bufferMaxLen) {
      buffer.pop();
      priority.pop();
    }
    buffer = [timeStep, ...buffer];
    let importance = maxPriority;
    priority = [importance, ...priority];

    batch = [];
    while (true) {
      let index;
      index = Math.floor(Math.random() * buffer.length);
      if (prioritized && priority[index] < 3) {
        continue;
      } else {
      }

      if (batch.indexOf(index) === -1) {
        batch.push(index);
      }
      if (batch.length >= batchLen) {
        break;
      }
    }
    batch.forEach((idx) => {
      let reducePriority = Math.floor(Math.random() * buffer.length);
      priority[idx] = Math.max(priority[idx] - reducePriority, 0);
    });
  }
</script>

<svg viewBox="0 0 {width} {height}">
  <!-- CONTAINER -->
  <rect
    fill="none"
    stroke="var(--text-color)"
    x={bufferLeftStart - containerGap}
    y={containerY - containerGap}
    width={bufferMaxLen * (boxSize + gap) + containerGap}
    height={boxSize + priorityLen * maxPriority + containerGap * 2}
  />
  <!-- LINES -->
  {#each batch as experience, idx}
    <line
      stroke="var(--text-color)"
      stroke-width="0.5px"
      x1={batchLeftStart + boxSize / 2 + idx * (boxSize + gap)}
      y1={yBatch + boxSize}
      x2={bufferLeftStart + boxSize / 2 + experience * (boxSize + gap)}
      y2={y - priority[experience] * priorityLen}
    />
  {/each}
  <!-- BATCH -->
  <text
    stroke="none"
    fill="var(--text-color)"
    x={width / 2}
    y={15}
    dominant-baseline="middle"
    text-anchor="middle"
  >
    BATCH
  </text>
  {#each batch as experience, idx}
    <rect
      stroke="var(--text-color)"
      fill="var(--background-color)"
      x={batchLeftStart + idx * (boxSize + gap)}
      y={yBatch}
      width={boxSize}
      height={boxSize}
    />
    <text
      stroke="none"
      fill="var(--text-color)"
      x={batchLeftStart + idx * (boxSize + gap) + boxSize / 2}
      y={yBatch + boxSize / 2}
      dominant-baseline="middle"
      text-anchor="middle"
    >
      {buffer[experience]}
    </text>
  {/each}

  <!-- MEMORY BUFFER -->
  <text
    stroke="none"
    fill="var(--text-color)"
    x={width / 2}
    y={height - 15}
    dominant-baseline="middle"
    text-anchor="middle"
  >
    MEMORY BUFFER
  </text>
  {#each buffer as experience, idx}
    <rect
      stroke="black"
      fill="var(--main-color-1)"
      x={bufferLeftStart + idx * (boxSize + gap)}
      y={y - priority[idx] * priorityLen}
      width={boxSize}
      height={boxSize + priority[idx] * priorityLen}
    />
    <text
      stroke="none"
      fill="var(--background-color)"
      x={bufferLeftStart + idx * (boxSize + gap) + boxSize / 2}
      y={y + boxSize / 2}
      dominant-baseline="middle"
      text-anchor="middle"
    >
      {experience}
    </text>
  {/each}
</svg>
<div class="flex-center">
  <Button on:click={handleClick} value={"Step"} />
</div>
