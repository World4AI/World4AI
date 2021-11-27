<script>
  export let type = 'Integer';
  export let value = 42;
  export let count = 1;
</script>

<table>
  <thead>
    <tr>
      <th>PyObject</th>
    <tr>
  </thead>
  <tbody>
    <tr>
      <td>Type</td>
      <td>{type}</td>
    </tr>
    <tr>
      <td>Value</td>
      <td>{value}</td>
    </tr>
    <tr>
      <td>Count</td>
      <td>{count}</td>
    </tr>
  </tbody>
</table>

<style>
  table {
   width: 280px;
  }

  th {
   text-transform: uppercase;
  }

  td, th {
   border: 1px solid var(--text-color);
   padding: 5px;
   text-align: center;
  }
</style>
