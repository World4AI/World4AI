<script>
  import Container from "$lib/Container.svelte";
  // can be info, warning and danger 
  export let type;
  let background;

  if (type === "info") {
      background = "var(--main-color-4)";
  } else if (type === "warning") {
      background = "var(--main-color-3)";
  } else if (type === "danger") {
      background = "var(--main-color-1)";
  }

  let color = type === "danger" ? "white" : "black";
</script>

<Container>
  <div class="flex" style:background style:color>
    <div class="icon">
      <!-- we use feather icons -->
      {#if type === "info"}
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-info">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      {:else if type === "warning"}
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-alert-triangle">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
          <line x1="12" y1="9" x2="12" y2="13"></line>
          <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
      {:else if type === "danger"}
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-alert-octagon">
          <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
          <line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      {/if}
    </div>
    <p class="content">
      <slot />
    </p>
  </div>
</Container>

<style>
  .flex {
      display: flex;
      align-items: center;
      justify-content: flex-start;
      gap: 10px;
      padding: 15px 20px;
      border-radius: 7px;
  }

  .content {
      margin: 0;
      padding: 0;
    }
</style>
