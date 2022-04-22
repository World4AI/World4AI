<script>
  import Highlight from "svelte-highlight";
  import python from "svelte-highlight/languages/python-repl";
  import "svelte-highlight/styles/nord.css";

  export let code;
</script>

<div>
  <Highlight language={python} {code} />
</div>

<style>
  div {
    overflow: auto;
    max-width: 90%;
  }
</style>
