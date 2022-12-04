<script>
  export let header = [];
  export let data = [];
</script>

<div class="overflow-auto rounded-lg shadow">
  <table class="w-full">
    <thead class="bg-gray-50 border-b-2 border-gray-200">
      <tr>
        {#each header as entry}
          <th class="p-3 text-sm text-left font-semibold tracking-wide"
            >{entry}</th
          >
        {/each}
      </tr>
    </thead>
    <tbody class="divide-y divide-gray-100">
      {#each data as row}
        <tr>
          {#each row as cell}
            <td class="p-3 text-sm text-gray-700 whitespace-nowrap">{cell}</td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
