<script>
  export let header = [];
  export let data = [];
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
    {#each data as row}
      <tr>
        {#each row as cell}
          {#if typeof cell !== "object"}
            <td>{cell}</td>
          {:else}
            <td>
              {#each cell as element}
                <div>
                  Probability: {element.probability}, State:{element.state},
                  Reward:{element.reward}
                </div>
              {/each}
            </td>
          {/if}
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
</style>

