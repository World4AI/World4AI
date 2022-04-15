<script>
  import Question from "$lib/Question.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | History</title>
  <meta
    name="description"
    content="Deep learning frameworks are necessary for an efficient workflow. At the moment TensorFlow, PyTorch and JAX dominate the market."
  />
</svelte:head>

<h1>Deep Learning Frameworks</h1>
<Question>What deep learning frameworks are available?</Question>
<div class="separator" />
<p>
  Deep learning algorithms are usually implemented with specialized deep
  learning frameworks. These frameworks can run on the GPU to improve the speed
  of learning, have build in functionality that is commonly used in deep
  learning research and are battle tested. Implementing all those details by
  hand is usually a bad idea, unless you do it for learning purposes.
</p>
<p>
  Deep learning frameworks have a rich history. If we wrote this chapter two
  years ago the content would look different. Five years ago the contents would
  be different again. That says a lot about the speed with which the development
  of those frameworks progresses.
</p>
<p>
  All of the below mentioned frameworks use Python as the interface between the
  programmer and the underlying execution. There is usually a way to write your
  code in C++, because the underlying code is written in C++, but mostly you are
  out of luck if you prefer any other programming language. Python is king in
  the deep learning community.
</p>

<h2>
  TensorFlow

  <svg
    width="32px"
    height="32px"
    viewBox="0 0 32 32"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      stroke="var(--text-color)"
      fill="none"
      d="M26.135 16l0.13 6.266-4.13-2.401v8.938l-5.469 3.198v-32l13.599 7.865v7.068l-8.13-4.797v3.599zM1.734 7.865l13.599-7.865v32l-5.469-3.198v-18.667l-8.13 4.797z"
    />
  </svg>
  and Keras
  <svg
    width="32px"
    height="32px"
    viewBox="0 0 24 24"
    role="img"
    xmlns="http://www.w3.org/2000/svg"
    ><title>Keras icon</title><path
      stroke="var(--text-color)"
      fill="none"
      d="M24 0H0v24h24V0zM8.45 5.16l.2.17v6.24l6.46-6.45h1.96l.2.4-5.14 5.1 5.47 7.94-.2.3h-1.94l-4.65-6.88-2.16 2.08v4.6l-.19.2H7l-.2-.2V5.33l.17-.17h1.48z"
    /></svg
  >
</h2>
<p>
  TensorFlow is developed by Google and was first released in the year 2015.
  That makes the framework the oldest in the list. Keras is a higher level api
  that sits on top of TensorFlow. Keras was developed independently by François
  Chollet in order to facilitate fast and easy deep learning development.
  Originally Keras supported multiple deep learning libraries (for example
  Theano), but became an integral part of TensorFlow 2 in 2019 and nowadays only
  supports TensorFlow. You can say that TensorFlow and Keras go hand in hand.
  Keras is an incredible great choice for beginners. It has a vibrant community
  and a mass of tutorials and books to learn from.
</p>
<div class="separator" />

<h2>
  PyTorch
  <svg
    width="32px"
    height="32px"
    viewBox="0 0 24 24"
    role="img"
    xmlns="http://www.w3.org/2000/svg"
    ><title>PyTorch icon</title><path
      stroke="var(--text-color)"
      fill="none"
      d="M12.005.04l-7.03 7.03a9.832 9.832 0 0 0 0 13.975 9.833 9.833 0 0 0 13.976 0c3.97-3.887 3.972-10.171.084-13.976l-1.738 1.737c2.895 2.895 2.895 7.608 0 10.503-2.894 2.894-7.608 2.894-10.503 0C3.9 16.414 3.9 11.7 6.794 8.806l4.632-4.631.58-.663zm3.556 3.886a1.323 1.323 0 0 0-1.323 1.323 1.323 1.323 0 0 0 1.323 1.323 1.323 1.323 0 0 0 1.323-1.323 1.323 1.323 0 0 0-1.323-1.323z"
    /></svg
  >
</h2>
<p>
  PyTorch is developed by Facebook and was initially released in 2016. PyTorch
  was inspired by Torch, a lua based deep leanring library. In the years after
  the initial release PyTorch has overtaken TensorFlow in terms of the research
  papers that are written with a particualr framework. PyTorch feels more
  intuitive to Python developers, you can say that it is more pythonic.
</p>
<div class="separator" />

<h2>JAX</h2>
<p>
  JAX is the new kid on the block. Like Keras and TensorFlow, JAX is developed
  by google. The first release appeard in the year 2020 on github. JAX has been
  gaining a lot of traction over the last years and month.
</p>
<div class="separator" />

<p>
  At the moment of writing most of our code examples are implemented in PyTorch.
</p>
<div class="separator" />
