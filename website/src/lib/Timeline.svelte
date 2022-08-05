<script>
  //format: { date: 2006, title: "Title", description: "description" },
  export let timeline = [];
</script>

<div class="timeline">
  {#each timeline as event}
    <div class="event">
      <p class="date">{event.date}</p>
      <h3 class="title">{event.title}</h3>
      <p class="description">{event.description}</p>
    </div>
  {/each}
</div>

<style>
  .timeline {
    max-width: 300px;
    margin: 0 auto;
    position: relative;
  }

  .timeline::before {
    position: absolute;
    content: "";
    background-color: var(--text-color);
    margin-left: -20px;
    top: 0;
    left: 0;
    width: 2px;
    height: 100%;
  }

  .event {
    padding: 0;
    margin-bottom: 10px;
    position: relative;
  }

  .event::before {
    position: absolute;
    content: "";
    width: 20px;
    height: 20px;
    background-color: var(--background-color);
    top: -5px;
    left: 0;
    margin-left: -25px;
  }

  .event::after {
    position: absolute;
    content: "";
    width: 10px;
    height: 10px;
    background-color: var(--text-color);
    top: 0;
    left: 0;
    margin-left: -24px;
  }

  .date {
    margin: 0;
    background-color: var(--text-color);
    color: var(--background-color);
    font-size: 12px;
    font-weight: bold;
    width: fit-content;
    text-align: center;
    padding: 5px 10px;
  }

  .title {
    text-align: start;
    margin: 10px 0 0 0;
    font-size: 20px;
  }

  .description {
    margin: 10px 0 0 0;
  }
</style>
