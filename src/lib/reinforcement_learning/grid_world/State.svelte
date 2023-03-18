<script>
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  export let state;
</script>

<Table>
  <TableHead>
    <Row>
      <HeaderEntry>Column</HeaderEntry>
      <HeaderEntry>Row</HeaderEntry>
    </Row>
  </TableHead>
  <TableBody>
    {#if state}
      <Row>
        <DataEntry>
          <span class="inline-block bg-red-100 px-5 py-1 rounded-full">
            {state.c}
          </span>
        </DataEntry>
        <DataEntry>
          <span class="inline-block bg-blue-100 px-5 py-1 rounded-full">
            {state.r}
          </span>
        </DataEntry>
      </Row>
    {/if}
  </TableBody>
</Table>
