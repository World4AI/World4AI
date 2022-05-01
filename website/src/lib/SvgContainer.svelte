<script>
  export let maxWidth;
</script>

<div class="container">
  <div style="max-width: {maxWidth};" class="svg-container">
    <slot />
  </div>
</div>

<style>
  .svg-container {
    margin: 0 auto;
  }
</style>
