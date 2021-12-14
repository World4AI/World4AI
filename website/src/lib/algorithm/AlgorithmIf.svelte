<div>
  <span class="algo-if">If <slot name="condition" /></span>
  <span class="algo-level"><slot /></span>
  <span class="algo-if">End If</span>
</div>

<style>
  div {
    padding-left: 5px;
    border-left: 1px dotted rgba(255, 255, 255, 0.2);
  }

  span {
    display: block;
  }

  .algo-level {
    margin-left: 30px;
    font-size: 15px;
  }

  .algo-if {
    font-size: 20px;
  }
</style>
