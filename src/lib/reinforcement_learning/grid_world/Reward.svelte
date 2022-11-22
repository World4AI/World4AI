<script>
  export let reward;
</script>

<table>
  <thead>
    <tr>
      <th>Reward</th>
    </tr>
  </thead>
  <tbody>
    {#if reward !== null}
      <tr>
        <td>{reward}</td>
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
