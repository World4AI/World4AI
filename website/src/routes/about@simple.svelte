<script>
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | About</title>
  <meta
    name="description"
    content="World4AI is used as a personalized plattform to learn and advance in artificial intelligence."
  />
</svelte:head>

<h1>What is World4AI?</h1>
<p class="warning">
  At the moment only a few sections have been written. The whole website is a
  work in progress.
</p>
<div class="separator" />
<p>
  World4AI is my answer to the question “How can I push my knowledge in
  artificial intelligence to an intermediary level?”. The main purpose of
  World4AI is for me to have a personalized blog that I can utilize to learn
  about AI. The blog is going to be structured in form of books, where knowledge
  is going to be organized in different chapters and different books.
</p>
<div class="separator" />

<p>
  This website is a centerpiece to a collection of books. The section called <Highlight
    >blocks</Highlight
  > contains the theory and intuition for many branches of artificial intelligence
  and mathematics. The section called <Highlight>books</Highlight>
  contains links to books written with the help of jupyter notebooks. This separation
  has a didactic advantage. A classical website written with html, css and javascript
  allows for interactive examples, that are hard to implement with jupyter notebooks.
  The notebooks on the other hand written in python allow to write practical python
  examples and to publish them directly, without the need to copy the results into
  hmtl. The intention is to build up the knowledge using the website and to practice
  the theory examples written in Python.
</p>
<div class="flex-center">
  <img width="50px" src="./logo/logo.svg" alt="World4AI Logo" />
</div>
<p>
  The logo of World4AI represents this idea of different blocks that come
  together to form a knowledge base.
</p>
<div class="separator" />

<style>
  .flex-center {
    display: flex;
    justify-content: center;
  }
</style>
