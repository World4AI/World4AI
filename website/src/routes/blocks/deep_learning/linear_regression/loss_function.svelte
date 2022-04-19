<script>
  import Question from "$lib/Question.svelte";
  import Scatterplot from "$lib/Scatterplot.svelte";

  let data = [
    [
      { x: 5, y: 5 },
      { x: 9, y: 9 },
      { x: 15, y: 19 },
      { x: 20, y: 25 },
      { x: 22, y: 15 },
      { x: 30, y: 22 },
      { x: 35, y: 20 },
      { x: 38, y: 30 },
      { x: 40, y: 45 },
      { x: 45, y: 25 },
    ],
  ];
</script>

<h1>Loss Function</h1>
<Question>How do we measure success and failure?</Question>
<div class="separator" />

<p>
  While we can intuitively tell how far away our line is from some optimal
  location, we need a quantitative measure that the linear regression algorithm
  can use for optimization purposes. In machine learning we use a measure called
  loss (or loss function). Using this quantity we can tweak the weights and
  biases to find a minimum value of the loss.
</p>
<Scatterplot
  {data}
  minX={0}
  maxX={50}
  mixY={0}
  maxY={50}
  numTicks={6}
  xLabel={"Feature"}
  yLabel={"Label"}
/>

<p>Squaring 1) non negative 2) outliers</p>
