<script>
  export let algoName;
</script>

<div class="algorithm">
  <div class="algo-top">
    <span class="algo-designation">Algorithm:</span>
    <span class="algo-name">{algoName}</span>
  </div>
  <span class="algo-content"><slot /></span>
</div>

<style>
  .algorithm {
    border-bottom: 1px solid var(--text-color);
  }

  .algo-content {
    font-size: 20px;
  }

  .algo-name {
    font-style: italic;
    font-size: 16px;
  }

  .algo-designation {
    font-weight: bold;
    text-transform: uppercase;
  }

  .algo-top {
    padding: 2px 0;
    margin-top: 0;
    border-top: 1px solid var(--text-color);
    border-bottom: 1px solid var(--text-color);
  }
</style>
