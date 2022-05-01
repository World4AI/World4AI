<script>
  import Highlight from "$lib/Highlight.svelte";
  import Sentence from "$lib/Sentence.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
</script>

<svelte:head>
  <title>World4AI | Books</title>
  <meta
    name="description"
    content="Jupyter notebooks contain implementations of the theory that is covered in the blocks section. The books contain topics such as Python, PyTorch, NumPy, Matplotlib, Pandas and so on."
  />
</svelte:head>

<h1>Jupyter Books</h1>
<div class="separator" />

<p>
  The Jupyter Books that are presented below are designed to provide an
  opportunity to practice the theory that was developed in the <Highlight
    >blocks</Highlight
  > section.
</p>
<SvgContainer maxWidth={"700px"}>
  <Sentence sentence={"jupyter"} />
</SvgContainer>
<SvgContainer maxWidth={"500px"}>
  <Sentence sentence={"books"} />
</SvgContainer>
<div class="separator" />
