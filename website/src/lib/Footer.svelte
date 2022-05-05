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
    <section class="notes">
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
    </section>
    <section class="references">
      {#if references.length > 0}
        <h4>References</h4>
        <ol class="references">
          {#each references as reference, idx}
            <li id="reference-{idx + 1}">
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
    </section>
  </footer>
</Container>
<div class="separator" />

<style>
  footer {
    padding: 20px 20px;
  }

  .notes {
    margin-bottom: 15px;
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
