<script>
  export let maxWidth = "1000px";
</script>

<div class="container" style="max-width: {maxWidth}">
  <slot />
</div>

<style>
  .container {
    margin: 0 auto;
    padding: 0 30px;
  }
</style>
