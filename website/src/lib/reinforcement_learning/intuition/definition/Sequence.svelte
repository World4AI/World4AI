<script>
  import { onMount } from "svelte";
  export let sequenceLength;

  let length = 0;
  let pointer = 0;
  onMount(() => {
    let interval = setInterval(() => {
      pointer += 1;
      length = (pointer % sequenceLength) + 1;
    }, 500);

    return () => clearInterval(interval);
  });

  //TODO, take the colors from css
  const color_1 = "#FF683C";
  const color_2 = "#4EB6D7";

  function createSequence(length) {
    let sequence = [];
    for (let i = 0; i < length; i++) {
      if (i === sequenceLength - 1) {
        sequence.push(color_2);
      } else {
        sequence.push(color_1);
      }
    }
    return sequence;
  }
  $: sequence = createSequence(length);
</script>

<svg viewBox="0 0 500 20">
  {#each sequence as step, idx}
    <rect
      height="20"
      width="20"
      x={idx * (20 + 2)}
      y="0"
      fill={step}
      stroke="black"
    />
  {/each}
</svg>

