<script>
  export let width = 100;
  export let height;
  export let maxWidth = "300px";
  export let components;
  export let arrowStrokeWidth=1;

  // show border for debugging
  export let debug = false;

  import SvgContainer from "$lib/SvgContainer.svelte";

  // imports for the diagram
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Plus from "$lib/diagram/Plus.svelte";
  import Circle from "$lib/diagram/Circle.svelte";
</script>

<SvgContainer {maxWidth}>
  <svg class:debug viewBox="0 0 {width} {height}">
    {#each components as component}
      {#if component.type === "block"}
        <Block
          text={component.text}
          fontSize={component.fontSize ? component.fontSize : 7}
          x={component.x}
          y={component.y}
          width={component.width}
          height={component.height}
          color={component.color ? component.color : "none"}
        />
      {:else if component.type === "circle"}
        <Circle 
          x={component.x}
          y={component.y}
          r={component.r}
          color={component.color}
        />
      {:else if component.type === "arrow"}
        <Arrow
          data={component.data}
          dashed={component.dashed}
          moving={component.moving}
          color={component.color}
	  strokeWidth={arrowStrokeWidth}
        />
      {:else if component.type === "plus"}
        <Plus x={component.x} y={component.y} />
      {/if}
    {/each}
  </svg>
</SvgContainer>

<style>
  .debug {
    border: 1px solid black;
  }
</style>
