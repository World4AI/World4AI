<script>
  export let maxWidth;
</script>

<div class="w-full">
  <div style="max-width: {maxWidth};" class="my-0 mx-auto">
    <slot />
  </div>
</div>
