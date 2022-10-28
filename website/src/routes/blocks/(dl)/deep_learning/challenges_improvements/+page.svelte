<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Challenges and Improvements</title>
  <meta
    name="description"
    content="Training neural networks can sometimes be a challenging problems. Knowing what types of problems you might encounter and techniques to deal with them is a necessary skill for machine learning engineers."
  />
</svelte:head>

<h1>Challenges and Improvements</h1>
<div class="separator" />

<Container>
  <p>
    The "circular data" example from the previous chapter was trivial to solve.
    In practice it might not be an easy task to make neural networks learn
    though. You could (and most likely will) experience a plethora of problems,
    which could make the training process a misearble experience, expecially if
    you can't find the source of the problem or don't know how to deal with it.
  </p>
  <p>
    In this chapter we will outline some common pitfalls and discuss approaches
    to deal with those. We will often demonstrate solutions to those problems
    using
    <a href="http://yann.lecun.com/exdb/mnist/" target="_blank" rel="noreferrer">MNIST</a> (Modified
    National Institute of Standards and Technology database). While some of the improvements
    we are going to cover are not strictly necessary to get good results for MNIST,
    using MNIST at this point in our deep learning journey will facilitate the explanations.
  </p>
  <div class="separator" />
</Container>
