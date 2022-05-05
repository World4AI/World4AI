<script>
  export let maxWidth;
</script>

<div class="container">
  <div style="max-width: {maxWidth};" class="svg-container">
    <slot />
  </div>
</div>

<style>
  .container {
    width: 100%;
  }
  .svg-container {
    margin: 0 auto;
  }
</style>
