<script>
  import katex from "katex";

  let html;
  function generateLatex(node) {
    html = katex.renderToString(node.textContent, {
      throwOnError: false,
    });
  }
</script>

<span class="placeholder" use:generateLatex><slot /></span>
<span>{@html html}</span>

<style>
  .placeholder {
    display: none;
  }
</style>
