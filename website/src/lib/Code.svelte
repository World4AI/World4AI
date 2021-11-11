<script>
  import Highlight from "svelte-highlight";
  import python from "svelte-highlight/src/languages/python";
  import "svelte-highlight/src/styles/gruvbox-dark-hard.css";

  export let code;
</script> 


<Highlight language="{python}" code="{code}" />
