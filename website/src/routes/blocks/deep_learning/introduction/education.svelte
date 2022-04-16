<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";

  let books = [
    [
      ["Author", "Sebastian Raschka"],
      ["Title", "Machine Learning with PyTorch and Scikit-Learn"],
      ["Level", "Beginner"],
      [
        "Description",
        "A good machine learnning book for beginners. The book covers classical machine leanring algorithms and moves on to deep learning with PyTorch.",
      ],
    ],
    [
      ["Author", "Aurélien Géron"],
      [
        "Title",
        "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow",
      ],
      ["Level", "Beginner"],
      [
        "Description",
        "If you have only one book to pick and do not know where to start, this is probabliy the right choice.",
      ],
    ],
    [
      ["Author", "François Chollet"],
      ["Title", "Deep Learning with Python, Second Edition"],
      ["Level", "Beginner"],
      [
        "Description",
        "This book is written by the creator of Keras, so you know you get quality content",
      ],
    ],
    [
      ["Author", "Lewis Tunstall, Leandro von Werra, Thomas Wolf"],
      ["Title", "Natural Language Processing with Transformers"],
      ["Level", "Advanced"],
      [
        "Description",
        "This book comes from the creators and employees of the Huggingface library. If you are interested in natural language processing, this is the right choice.",
      ],
    ],
  ];

  let courses = [
    [
      ["Name", "fast.ai"],
      ["Lecturer", "Jeremy Howard and Rachel Thomas"],
      ["Level", "Beginner"],
      ["Link", "fast.ai"],
      [
        "Description",
        "fast.ai is several things at the same time. It is a non-profit with the goal of democratizing deep learning, a deep learning library based on PyTorch, a book called 'Deep Learning for Coders With Fasai and Pytorch' and an online course that teaches you deep learning. The approach taken by fast.ai differs from most books and courses. They teach you how you can generate state of the art results first and go into the basic building blocks of neural networks second.",
      ],
    ],
    [
      ["Name", "Coursera"],
      ["Lecturer", "Andrew Ng"],
      ["Level", "Beginner"],
      [
        "Link",
        "https://www.deeplearning.ai/program/deep-learning-specialization/",
      ],
      [
        "Description",
        "This is probably the most famous and the most acknowledged course in deep learning. The course starts with the basic building blocks of deep learning and steady move towards state of the art. The course is not free, but there is financial support available if you can not afford the lectures.",
      ],
    ],
    [
      ["Name", "NYU Deep Learning"],
      ["Lecturer", "Yann LeCun and Alfredo Canziani"],
      ["Level", "Advanced"],
      ["Link", "https://cds.nyu.edu/deep-learning/"],
      [
        "Description",
        "This course is part of the deep learning lectures at NYU. The course is very indepth and very engaging, but is probably not the best choice for absolute beginners. We recommend you take this course after you have had some exposure to an intorductory course or book.",
      ],
    ],
  ];

  let channels = [
    [
      ["Channel", "Yannic Kilcher"],
      ["Level", "Advanced"],
      ["Link", "https://www.youtube.com/c/YannicKilcher"],
      [
        "Description",
        "Yannics channel has two important segments that are relevant to your machine learning journey. He has a weekly segement called 'ML News' where he discusses the newest develompents in the machine learning space. The second segment are his paper discussions, where he takes a state of the art deep learning paper apart and explains the concept at a granular level.",
      ],
    ],
    [
      ["Channel", "Machine Learning Street Talk"],
      ["Level", "Advanced"],
      ["Link", "https://www.youtube.com/c/MachineLearningStreetTalk"],
      [
        "Description",
        "This podcast invites AI experts to discuss either their life work or individual papers. The discussions are not easy to grasp, when you are a beginner.",
      ],
    ],
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Education</title>
  <meta
    name="description"
    content="Deep learning is a fast moving field. It is necessary to constantly improve your skills. You can use books, youtube channel and research papers to keep up to date."
  />
</svelte:head>

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
{#each books as book}
  <Table data={book} />
{/each}
<div class="separator" />

<h2>Online Courses</h2>
{#each courses as course}
  <Table data={course} />
{/each}
<div class="separator" />

<h2>YouTube Channels</h2>

{#each channels as channel}
  <Table data={channel} />
{/each}
<div class="separator" />

<h2>Deep Learning Practice</h2>
<p>
  You can not learn deep learning without applying the algorithms with a deep
  learning framework yourself. Learning the theory is not going to be
  sufficient. If you have the data and an idea, then go for it and upload your
  project to GitHub. A popular alternative is to compete in machine learning
  challenges. The most popular plattform that host competitions is <a
    href="https://www.kaggle.com"
    target="_blank">Kaggle</a
  >. There you can take part in discussions, upload datasets, create notebooks
  and take part in competitions. The plattform has a ranking system in the four
  mentioned categories and allows you to grow in the ranks and gain reputation
  and titles. Additionally many of the challenges give out monetary prices.
  Kaggle is essentially a social network, so if you want to get addicted to one,
  kaggle is a good choice for machine learning practicioners.
</p>
<div class="separator" />

<h2>Keeping Up With Research</h2>
<p>
  When you come from the outside of the deep learning research community, you do
  not have access to a network that you can utilize to find the most relevant
  research papers you should focus on. Fortunately you have access to twitter
  and can therefore follow companies that focus on deep learning research,
  established researchers and book authors to get access to your own individual
  network. Many of those sources will share interesting articles, tutorials,
  videos and the newest research. It will take some time to find which sources
  are the most reliable, but the benefits are enormous. We suggest that you
  create an account especially for the purpose of participating in the deep
  learning community and avoid other types of content. Otherwise you will
  experience the "dark side" of twitter.
</p>
<div class="separator" />
