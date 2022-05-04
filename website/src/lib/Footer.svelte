<script>
  import Container from "$lib/Container.svelte";
  // notes are just a list of strings
  export let notes = [];

  // references have a format similar to bibtex
  /* 
  {
      author: "",
      title: "",
      journal: "",
      year: "",
      pages: "",
      volume: "",
      issue: "",
  }
  */
  export let references = [];
</script>

<div class="separator" />
<Container>
  <footer>
    {#if notes.length > 0}
      <h4>Notes</h4>
      <ol class="notes">
        {#each notes as note, idx}
          <li id="note-{idx + 1}">
            {note}
            <a href="#note-origin-{idx + 1}">
              <!-- svg is from feather icons -->
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="feather feather-arrow-left-circle"
                ><circle cx="12" cy="12" r="10" /><polyline
                  points="12 8 8 12 12 16"
                /><line x1="16" y1="12" x2="8" y2="12" /></svg
              >
            </a>
          </li>
        {/each}
      </ol>
    {/if}

    {#if references.length > 0}
      <h4>References</h4>
      <ol class="references">
        {#each references as reference, idx}
          <li id="reference-{idx + 1}">
            {reference.author}.
            {reference.title}.
            {reference.journal}
            {reference.volume}
            ({reference.issue}),
            {reference.pages}, ({reference.year})
            <a href="#reference-origin-{idx + 1}">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="feather feather-arrow-left-circle"
                ><circle cx="12" cy="12" r="10" /><polyline
                  points="12 8 8 12 12 16"
                /><line x1="16" y1="12" x2="8" y2="12" /></svg
              >
            </a>
          </li>
        {/each}
      </ol>
    {/if}
  </footer>
</Container>
<div class="separator" />

<style>
  footer {
    padding: 20px 20px;
  }
  h4 {
    margin-bottom: 10px;
    font-weight: normal;
  }
  li {
    line-height: 1;
    font-size: 15px;
    opacity: 60%;
  }
  a {
    color: var(--text-color);
  }
</style>
