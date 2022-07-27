<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Sentence from "$lib/Sentence.svelte";
</script>

<svelte:head>
  <title>World4AI | Artificial Intelligence Blocks</title>
  <meta
    name="description"
    content="World4AI is a organized into different blocks. Each block represents a topic that is relevant to artificial intelligence. For example deep learning, symbolic ai or mathematics."
  />
</svelte:head>

<article>
  <h1>Blocks</h1>
  <div class="separator" />
  <Container>
    <p>
      World4AI is organized into building blocks. Each block builds knowledge
      that is fundamental to modern artificial intelligence. But you don't need
      to be proficient in any of the topics to start the next one and you do not
      need to learn the blocks in a stricly linear manner. You can learn the
      material that you require as you go along.
    </p>

    <SvgContainer maxWidth={"700px"}>
      <Sentence sentence={"blocks"} />
    </SvgContainer>
    <p>
      At the moment only a few sections have been started and none is considered
      finalized. More chapters will be published in the next weeks and months.
      But the overall process will most likely take many years.
    </p>
    <div class="separator" />
  </Container>
</article>

<style>
</style>
