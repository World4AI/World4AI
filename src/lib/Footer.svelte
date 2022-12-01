<script>
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
<footer class="container mx-auto">
  <section>
    {#if notes.length > 0}
      <h4 class="mb-2">Notes</h4>
      <ol class="mb-4">
        {#each notes as note, idx}
          <li id="note-{idx + 1}" class="text-sm leading-3 opacity-50 mb-3">
            {note}
            <a href="#note-origin-{idx + 1}" class="inline-block">
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
  </section>
  <section class="references">
    {#if references.length > 0}
      <h4 class="mb-2">References</h4>
      <ol class="references">
        {#each references as reference, idx}
          <li
            id="reference-{idx + 1}"
            class="text-sm leading-3 opacity-50 mb-3"
          >
            {#if reference.author}
              {reference.author}.
            {/if}
            {#if reference.title}
              {reference.title}.
            {/if}
            {#if reference.journal}
              {reference.journal}.
            {/if}
            {#if reference.volume}
              Vol. {reference.volume}.
            {/if}
            {#if reference.issue}
              ({reference.issue}).
            {/if}
            {#if reference.pages}
              pp. {reference.pages}.
            {/if}
            {#if reference.year}
              ({reference.year}).
            {/if}
            <a href="#reference-origin-{idx + 1}" class="inline-block">
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
  </section>
</footer>
<div class="separator" />
