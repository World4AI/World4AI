<script>
  export let value = "Button";
  export let disabled = false;
</script>

<button class:disabled {disabled} on:click>{value}</button>

<style>
  button {
    outline: none;
    border: 1px solid var(--main-color-1);
    width: fit-content;
    background-color: var(--background-color);
    color: var(--text-color);
    font-size: 20px;
    padding: 10px 20px;
    cursor: pointer;
    margin-bottom: 10px;
  }

  button:hover {
    color: var(--main-color-1);
  }

  .disabled {
    color: var(--main-color-1);
    border-color: var(--main-color);
    cursor: wait;
  }
</style>
