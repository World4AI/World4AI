<script>
  import katex from "katex";

  let html;
  function generateLatex(node) {
    html = katex.renderToString(node.textContent, {
      throwOnError: false,
    });
  }
</script>

<span class="hidden" use:generateLatex><slot /></span>
<span class="inline-block px-0 py-[2px] md:whitespace-nowrap">{@html html}</span
>

<svelte:head>
  <link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css"
    integrity="sha384-vKruj+a13U8yHIkAyGgK1J3ArTLzrFGBbBc0tDp4ad/EyewESeXE/Iv67Aj8gKZ0"
    crossorigin="anonymous"
  />
</svelte:head>
