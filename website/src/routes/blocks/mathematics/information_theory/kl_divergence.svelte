<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>KL Divergence</h1>
<Question>How is KL divergence defined?</Question>
<div class="separator" />

<p class="info">
  The KL divergence (also called relative entropy) measures the difference
  between two probability distributions.
</p>

<Latex
  >{String.raw`
\mathbb{E}_p\log \Big(\dfrac{p_{\theta}(x_i)}{q_{\phi}(x_i)}\Big)
= \sum_{i=1}^\infty p_{\theta}(x_i)\log \Big(\dfrac{p_{\theta}(x_i)}{q_{\phi}(x_i)}\Big) \\
`}</Latex
>
