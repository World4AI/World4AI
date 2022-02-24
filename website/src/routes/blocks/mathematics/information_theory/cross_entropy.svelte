<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Cross Entropy</h1>
<Question>How is cross entropy defined?</Question>
<div class="separator" />
<p />
<Latex
  >{String.raw`
H(p, q) = - \mathbb{E}_p[\log q(x)] = - \sum_x p(x) \log q(x)
  `}</Latex
>
