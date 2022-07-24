<script>
  import Container from "$lib/Container.svelte";
  import SvelteMarkdown from "svelte-markdown";
  import { marked } from "marked";
  import { Highlight } from "svelte-highlight";
  import python from "svelte-highlight/languages/python";
  import a11yDark from "svelte-highlight/styles/a11y-dark";
  import { onMount } from "svelte";
  import JupyterLatex from "$lib/JupyterLatex.svelte";

  export let url;
  let cells = [];

  onMount(async () => {
    console.log(url);
    const response = await fetch(url);
    const notebook = await response.json();

    notebook.cells.forEach((cell) => {
      if (cell.cell_type === "code") {
        cells.push({ type: cell.cell_type, content: cell.source.join("") });
        if (cell.outputs.length > 0) {
          if (cell.outputs[0].output_type === "stream") {
            cells.push({
              type: "result",
              content: cell.outputs[0].text.join(""),
            });
          } else if (cell.outputs[0].output_type === "execute_result") {
            cells.push({
              type: "result",
              content: cell.outputs[0].data["text/plain"].join(""),
            });
          } else if (cell.outputs[0].output_type === "display_data") {
            cells.push({
              type: "image",
              content: cell.outputs[0].data["image/png"],
            });
          }
        }
      } else if (cell.cell_type === "markdown") {
        cells.push({ type: cell.cell_type, content: cell.source.join("") });
      }
    });
    cells = cells;
  });

  //TODO there is probably a but in RegEx
  // logic to process latex
  const latexTokenizer = {
    name: "latex",
    level: "inline",
    start(src) {
      return src.indexOf("$");
    },
    tokenizer(src) {
      const rule = /^\$+([^\$\n]+?)\$+/;
      const match = rule.exec(src);
      if (match) {
        return {
          type: "latex",
          raw: match[0],
          text: match[1].trim(),
        };
      }
    },
  };
  marked.use({ extensions: [latexTokenizer] });
  const options = marked.defaults;

  const renderers = { latex: JupyterLatex };
</script>

<svelte:head>
  {@html a11yDark}
</svelte:head>

<Container>
  {#each cells as cell}
    {#if cell.type === "code"}
      <div class="code-container">
        <Highlight language={python} code={cell.content} />
      </div>
    {:else if cell.type === "result"}
      <div class="result-container">
        <pre>{cell.content}</pre>
      </div>
    {:else if cell.type === "markdown"}
      <SvelteMarkdown {options} {renderers} source={cell.content} />
    {:else if cell.type === "image"}
      <img src="data:image/png;base64,{cell.content}" alt="code output" />
    {/if}
  {/each}
  <div class="separator" />
</Container>

<style>
  .code-container {
    margin-bottom: 10px;
  }

  .result-container {
    padding: 10px;
    overflow-x: auto;
    border: 1px solid rgba(0, 0, 0, 0.2);
    margin: 10px 0;
  }

  pre {
    padding: 0;
  }

  img {
    display: block;
    margin: 0 auto;
    width: 100%;
    max-width: 500px;
  }
</style>
