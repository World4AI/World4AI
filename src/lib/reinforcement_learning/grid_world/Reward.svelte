<script>
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  export let reward;
</script>

<Table>
  <TableHead>
    <Row>
      <HeaderEntry>Reward</HeaderEntry>
    </Row>
  </TableHead>
  <TableBody>
    <Row>
      <DataEntry>
        <span class="bg-red-100 px-5 py-1 rounded-full">
          {reward}
        </span>
      </DataEntry>
    </Row>
  </TableBody>
</Table>
