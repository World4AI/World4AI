<script>
  export let state;
</script>

<table>
  <thead>
    <tr>
      <th>Column</th>
      <th>Row</th>
    </tr>
  </thead>
  <tbody>
    {#if state}
      <tr>
        <td>{state.c}</td>
        <td>{state.r}</td>
      </tr>
    {/if}
  </tbody>
</table>

<style>
  table {
    width: 100%;
    max-width: 200px;
  }

  th {
    text-transform: uppercase;
  }

  td,
  th {
    border: 1px double var(--text-color);
    padding: 7px;
    text-align: center;
  }
</style>
