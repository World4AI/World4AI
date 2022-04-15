<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";

  booksHeader = ["Author", "Title", "Level"];
  booksData = [
    [
      "Sebastian Raschka",
      "Machine Learning with PyTorch and Scikit-Learn",
      "Beginner",
    ],
    [
      "Aurélien Géron",
      "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow, Second Edition",
      "Beginner",
    ],
    [
      "François Chollet",
      "Deep Learning with Python, Second Edition",
      "Beginner",
    ],
  ];
  youtubeHeader = ["Channel", "Description"];
  youtubeData = [
    ["Yannic Kilcher", ""],
    ["Machine Learning Street Talk", ""],
  ];
</script>

<h1>Education in Deep Learning</h1>
<Question>How do you study and stay up to date with deep learning?</Question>
<div class="separator" />
<p>
  It usually requires more than one single resource to understand a topic. You
  should explore different books, videos, blog posts and research papers to get
  the full picture. Once you grasp the basics of deep learning, you will still
  need to find a way to keep up to date with the newest research and this might
  not be the easiest task. Below we attempt to list the most relevant literature
  for beginners and hints on ways you can stay informed regarding the newest
  developments.
</p>
<div class="separator" />

<h2>Books</h2>
<Table header={booksHeader} data={booksData} />
<div class="separator" />

<h2>YouTube Channels</h2>
<Table header={youtubeHeader} data={youtubeData} />
<div class="separator" />

<h2>Deep Learning Practice</h2>
<p>Kaggle and Github Projects</p>
<div class="separator" />

<h2>Keeping Up With Research</h2>
<p>Twitter</p>
<div class="separator" />
