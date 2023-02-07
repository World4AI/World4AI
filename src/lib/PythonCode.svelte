<script>
  import Highlight from "svelte-highlight";
  import python from "svelte-highlight/languages/python";
  import pythonRepl from "svelte-highlight/languages/python-repl";
  import github from "svelte-highlight/styles/base16-github";

  export let code = "";
  export let isOutput = false;
</script>

<svelte:head>
  {@html github}
</svelte:head>

<div class="border my-2 text-sm">
  {#if !isOutput}
    <Highlight language={python} {code} />
  {:else}
    <Highlight language={pythonRepl} {code} />
  {/if}
</div>
