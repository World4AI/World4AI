<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<SvgContainer maxWidth="500px">
  <svg viewBox="0 0 500 500">
    <circle cx="100" cy="100" r="50" fill="white" stroke="black" />

    <foreignObject x="50" y="50" width="100" height="100">
      <div xmlns="http://www.w3.org/1999/xhtml">
        <Latex xmlns="http://www.w3.org/1999/xhtml"
          >{String.raw`\dfrac{1}{2}`}</Latex
        >
      </div>
    </foreignObject>
  </svg>
</SvgContainer>

<style>
  div {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100px;
  }
</style>
