<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const width = 400;
  const height = 400;
  const blockWidth = 160;
  const blockHeight = 60;
  const offset = 5;
</script>

<SvgContainer maxWidth={"400px"}>
  <svg version="1.1" viewBox="0 0 {width} {height}">
    <Block
      x={width / 2}
      y={offset + blockHeight / 2}
      width={blockWidth}
      height={blockHeight}
      text="Agent"
      fontSize={23}
      class="fill-yellow-200"
    />
    <Block
      x={width / 2}
      y={height - offset - blockHeight / 2}
      width={blockWidth}
      height={blockHeight}
      text="Environment"
      fontSize={23}
      class="fill-green-200"
    />
    <Arrow
      data={[
        { x: width / 2 - blockWidth / 2, y: height - offset - blockHeight / 2 },
        {
          x: width / 2 - blockWidth / 2 - 70,
          y: height - offset - blockHeight / 2,
        },
        {
          x: width / 2 - blockWidth / 2 - 70,
          y: offset + blockHeight / 2,
        },
        {
          x: width / 2 - blockWidth / 2 - 10,
          y: offset + blockHeight / 2,
        },
      ]}
      dashed={true}
      strokeWidth={2.5}
      strokeDashArray="8 8"
      moving={true}
      speed={100}
    />
    <Arrow
      data={[
        {
          x: width / 2 + blockWidth / 2,
          y: offset + blockHeight / 2,
        },
        { x: width / 2 + blockWidth / 2 + 70, y: offset + blockHeight / 2 },
        {
          x: width / 2 + blockWidth / 2 + 70,
          y: height - offset - blockHeight / 2,
        },
        {
          x: width / 2 + blockWidth / 2 + 10,
          y: height - offset - blockHeight / 2,
        },
      ]}
      dashed={true}
      strokeWidth={2.5}
      strokeDashArray="8 8"
      moving={true}
      speed={100}
    />
    <Block
      x={width / 2 + blockWidth / 2 + 70}
      y={height / 2}
      width={35}
      height={35}
      class="fill-slate-200"
      type="latex"
      text={"A"}
      fontSize={22}
    />
    <Block
      x={width / 2 - blockWidth / 2 - 70 - 25}
      y={height / 2}
      width={35}
      height={35}
      class="fill-slate-200"
      type="latex"
      text={"S"}
      fontSize={22}
    />
    <Block
      x={width / 2 - blockWidth / 2 - 70 + 25}
      y={height / 2}
      width={35}
      height={35}
      class="fill-slate-200"
      type="latex"
      text={"R"}
      fontSize={22}
    />
  </svg>
</SvgContainer>
