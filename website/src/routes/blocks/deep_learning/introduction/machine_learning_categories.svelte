<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Table from "$lib/Table.svelte";

  let header = ["Location", "Size", "Price"];
  let data = [
    ["London", "100", 1_000_000],
    ["Berlin", "30", 80_000_000],
    ["...", "...", "..."],
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Machine Learning Categories</title>
  <meta
    name="description"
    content="Machine learning can be divided into different categories: supervised learning, unsupervised learning, self-supervised learning, semi-supervised learning and reinforcement learning."
  />
</svelte:head>

<h1>Categories of Machine Learning</h1>
<Question>What are the common categories of machine learning?</Question>
<div class="separator" />

<p>
  Machine learning is commonly divided into specific categories. The
  categorization usually depends on the amount of human supervision that is
  required for the algorithm to work.
</p>

<h2>Supervised Learning</h2>
<p>
  As the name supervised learning suggests, there is a human supervisor who
  labels the input data with the correct output that a machine learning
  algorithm should produce. The different inputs are called <Highlight
    >features</Highlight
  >, while the outputs are called <Highlight>labels</Highlight>.
</p>
<p>
  Let us for example assume that we want to estimate the price of a house based
  on the location and the size of the house. In that case the location and the
  size are the features, while the price is the label.
</p>
<Table {header} {data} />
<p>
  The two common tasks that we try to solve with supervised learning are
  <Highlight>regression</Highlight> and <Highlight>classification</Highlight>.
</p>
<p>
  In a classification task there is a finite number of classes that the machine
  learning algorithm needs to determine based on the labels. The usual example
  that is brought up in the machine learning literature is a spam filter. Based
  on the header and content of the email the algorithm needs to decide whether
  the email is ham or spam.
</p>
<p>
  In a regression task on the other hand the algorithm produces a continuous
  number. Predicting the price of the house based on labels is a regression
  task.
</p>
<div class="separator" />

<h2>Unsupervised Learning</h2>
<p>
  In unsupervised learning the dataset contains only features and no labels. The
  overall task is to find some hidden structure in the data. We could for
  example use the labels in the house pricing dataset and cluster those houses
  according to the features of the houses. Similar houses would be allocated to
  the same cluster, while different houses should be in vastly different
  clusters.
</p>
<div class="separator" />

<h2>Semi-supervised Learning</h2>
<p>
  Labeling your dataset is costly, therefore companies and researchers try to
  get away with labeling as few samples as possible. In semi-supervised learning
  only a fraction of data has labels, while the rest is unlabeled. The labeled
  data is used to train the algorithm and to label the remaining data. After
  that step the whole dataset can be used for training.
</p>
<div class="separator" />

<h2>Self-supervised Learning</h2>
<p>
  In self-supervised learning is in a a way supervised learning, yet the label
  is not determined by a human supervisor, but is derived directly from the
  features.
</p>
<div class="separator" />

<h2>Reinforcement Learning</h2>
<p>
  Reinforcemnt learning deals with sequential decisions, where an agent
  interacts with the environment and receives rewards based on the decisions. We
  have dedicated a whole block to reinforcement learning.
</p>
<div class="separator" />
