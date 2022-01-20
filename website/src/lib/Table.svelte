<script>
  export let header = [];
  export let data = [];
  export let idxHighlight = null;
</script>

<table>
  <thead>
    <tr>
      {#each header as entry}
        <th>{entry}</th>
      {/each}
    </tr>
  </thead>
  <tbody>
    {#each data as row, idx}
      <tr class:highlighted={idx === idxHighlight}>
        {#each row as cell}
          <td>{cell}</td>
        {/each}
      </tr>
    {/each}
  </tbody>
</table>

<style>
  table {
    width: 100%;
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
  td {
    font-style: italic;
  }

  .highlighted {
    color: var(--main-color-1);
  }
</style>
