<script>
  export let number;
  export let maxWidth = "130px";
  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";

  let width = number[0].length * 13 + 4;
  let height = number.length * 13 + 4;
</script>

<SvgContainer {maxWidth}>
  <svg viewBox = "0 0 {width} {height}">
    {#each number as row, rowIdx}
      {#each row as cell, colIdx}
        <Block x={8+colIdx*13} y={8+rowIdx*13} width={10} height={10} color={cell === 1 ? 'black' : "none"} />
      {/each}
    {/each}
  <svg>
</SvgContainer>
