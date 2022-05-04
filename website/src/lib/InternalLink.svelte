<script>
  export let type;
  export let id;
</script>

<sup>
  <a href="#{type}-{id}"
    ><sup id="{type}-origin-{id}">
      [{id}]
    </sup></a
  >
</sup>

<style>
  sup {
    scroll-margin-top: 100px;
    vertical-align: 2px;
  }

  a {
    color: var(--text-color);
    font-size: 15px;
    text-decoration: none;
    font-weight: bold;
  }
</style>
