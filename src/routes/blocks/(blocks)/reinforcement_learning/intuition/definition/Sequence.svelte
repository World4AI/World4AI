<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  export let sequenceLength;

  let sequence = [];
  for (let i = 0; i < sequenceLength; i++) {
    sequence.push(i);
  }

  const boxSize = 20;
</script>

<SvgContainer maxWidth={"500px"}>
  <svg viewBox="0 0 370 22">
    {#each sequence as _, idx}
      <Block
        x={1 + idx * boxSize + boxSize / 2 + idx * 4}
        y={1 + boxSize / 2}
        width={boxSize}
        height={boxSize}
        text={idx + 1}
        fontSize={12}
        class={idx === sequenceLength - 1 ? "fill-blue-400" : "fill-red-400"}
      />
    {/each}
  </svg>
</SvgContainer>
