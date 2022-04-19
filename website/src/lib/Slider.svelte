<script>
  export let value;
  export let min;
  export let max;
  export let step = 1;
</script>

<input type="range" bind:value {min} {max} {step} />

<style>
  input[type="range"] {
    height: 36px;
    -webkit-appearance: none;
    margin: 10px 0;
    width: 100%;
    background-color: var(--background-color);
  }
  input[type="range"]:focus {
    outline: none;
  }
  input[type="range"]::-webkit-slider-runnable-track {
    width: 100%;
    height: 12px;
    cursor: pointer;
    box-shadow: 1px 1px 1px #000000;
    background: var(--text-color);
    border: 1px solid #000000;
  }
  input[type="range"]::-webkit-slider-thumb {
    border: 1px solid #000000;
    height: 28px;
    width: 18px;
    background: var(--main-color-1);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -9px;
  }
  input[type="range"]:focus::-webkit-slider-runnable-track {
    background: var(--text-color);
  }
  input[type="range"]::-moz-range-track {
    width: 100%;
    height: 12px;
    cursor: pointer;
    box-shadow: 1px 1px 1px #000000;
    background: var(--text-color);
    border: 1px solid #000000;
  }
  input[type="range"]::-moz-range-thumb {
    border: 1px solid #000000;
    height: 28px;
    width: 18px;
    background: var(--main-color-1);
    cursor: pointer;
  }
  input[type="range"]::-ms-track {
    width: 100%;
    height: 12px;
    cursor: pointer;
    background: transparent;
    border-color: transparent;
    color: transparent;
  }
  input[type="range"]::-ms-fill-lower {
    background: var(--text-color);
    border: 1px solid #000000;
    box-shadow: 1px 1px 1px #000000;
  }
  input[type="range"]::-ms-fill-upper {
    background: var(--text-color);
    border: 1px solid #000000;
    box-shadow: 1px 1px 1px #000000;
  }
  input[type="range"]::-ms-thumb {
    margin-top: 1px;
    border: 1px solid #000000;
    height: 28px;
    width: 18px;
    background: var(--main-color-1);
    cursor: pointer;
  }
  input[type="range"]:focus::-ms-fill-lower {
    background: var(--text-color);
  }
  input[type="range"]:focus::-ms-fill-upper {
    background: var(--text-color);
  }
</style>
