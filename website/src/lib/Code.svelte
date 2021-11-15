<script>
  import Highlight from "svelte-highlight";
  import python from "svelte-highlight/src/languages/python-repl";
  import "svelte-highlight/src/styles/nord.css";

  export let code;
</script> 


<Highlight language="{python}" code="{code}" />
