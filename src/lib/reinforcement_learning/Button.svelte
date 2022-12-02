<script>
  export let value = "Button";
  export let disabled = false;
</script>

<button class:disabled {disabled} on:click>{value}</button>

<style>
  button {
    outline: none;
    border: 1px solid var(--text-color);
    width: fit-content;
    height: fit-content;
    background-color: var(--main-color-3);
    color: var(--text-color);
    font-size: 20px;
    padding: 10px 20px;
    cursor: pointer;
    margin-bottom: 10px;
    border-radius: 8px;
    font-weight: 600;
  }

  button:hover {
    transform: scale(1.02);
  }

  .disabled {
    color: var(--main-color-1);
    border-color: var(--main-color);
    cursor: wait;
  }
</style>
