<script>
  // for reference the following codepen was used
  // https://codepen.io/tippingpointdev/pen/bGgLqLY?editors=0110
  export let value;
  export let min;
  export let max;
  export let showValue = false;
  export let step = 1;
  export let label = null;
  export let labelId = '';
  $: percent = `${(value - min) / (max - min) * 100}%`;
</script>

{#if label}
  <label for={labelId}>{label}</label>
  {#if showValue}
  <span class="inline-block bg-blue-100 px-2 rounded-lg">{value}</span>
  {/if}
{/if}
<input id={labelId} class="inline-block align-middle" style:background-size='{percent} 100%' type="range" bind:value {min} {max} {step} />

<style>
  input[type="range"] {
    appearance: none;
    -webkit-appearance: none;
    margin-right: 15px;
    width: 100%;
    height: 8px;
    background: var(--main-color-4);
    border-radius: 5px;
    background-image: linear-gradient(var(--main-color-1), var(--main-color-1));
    background-repeat: no-repeat;
    border: 2px solid black;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 25px;
    width: 25px;
    border-radius: 50%;
    background: white;
    border: 2px solid black;
    cursor: ew-resize;
    box-shadow: 0 0 2px 0 #555;
    transition: background .3s ease-in-out;
  }
  
  input[type="range"]::-moz-range-thumb {
    appearance: none;
    -webkit-appearance: none;
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--main-color-1);
    border: 1px solid black;
    cursor: ew-resize;
    box-shadow: 0 0 2px 0 #555;
    transition: background .3s ease-in-out;
  }


input[type="range"]::-ms-thumb {
    appearance: none;
    -webkit-appearance: none;
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--main-color-1);
    border: 1px solid black;
    cursor: ew-resize;
    box-shadow: 0 0 2px 0 #555;
    transition: background .3s ease-in-out;
}
</style>
