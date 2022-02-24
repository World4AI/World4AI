<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Entropy</h1>
<Question>What is entropy?</Question>
<div class="separator" />

<p class="info">The entropy is defined as the expected level of information.</p>

<Latex
  >{String.raw`
H(x) = \sum_x -p(x) * log_2(p(x))
  `}</Latex
>
