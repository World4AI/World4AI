<script>
  import Container from "$lib/Container.svelte";
  import Blackboard from "../_education/Blackboard.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Education</title>
  <meta
    name="description"
    content="Deep learning is a fast moving field. It is necessary to constantly improve your skills. You can use books, youtube channel and research papers to keep up to date."
  />
</svelte:head>

<h1>Education in Deep Learning</h1>
<div class="separator" />
<Container>
  <SvgContainer maxWidth={"600px"}>
    <Blackboard />
  </SvgContainer>
  <p>
    Before we take a deep dive into the foundations of deep learning we would
    like to talk about additional learning resources and to encourage you to
    consult many different sources on your deep learning journey. It usually
    requires more than one single resource to understand a topic. You should
    explore different books, videos, blog posts and research papers to get the
    full picture. Once you grasp the basics of deep learning, you still need to
    find a way to keep up to date with the newest research and this might not be
    the easiest task. Below we attempt to list the most relevant literature for
    beginners and give hints on ways you can stay informed regarding the newest
    developments in deep learning.
  </p>
  <div class="separator" />

  <h2>Books</h2>
  <p>
    Books are a great gateway into the world of deep learning. Especially when
    it comes to beginner level books the choices are great.
  </p>

  <div class="book-container">
    <img
      src="/books/geron_book.jpeg"
      alt="Book Cover Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow"
    />
    <p>
      "Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow" by
      Aurélien Géron is a great choice, especially if you don't know where to
      start. If you have only one book to pick, this is probabliy the right
      pick. At the moment of writing the second edition is available in online
      stores, but the third edition is currently being written and is due
      somewhere at the end of 3rd or the beginning the 4th quarter of 2022. The
      book covers classical machine learning and goes on to discuss deep
      learning with Keras and TensorFlow in detail. While the book is very well
      suited for beginners, advanced topics like handling large amounts of data
      are covered as well.
    </p>
  </div>
  <div class="book-container">
    <img
      src="/books/cholet_book.png"
      alt="Book Cover DEEP LEARNING with Python"
    />
    <p>
      The book "DEEP LEARNING with Python" is written by the creator of Keras,
      so you know you get quality content. François Chollet covers exclusively
      deep learning with a high focus on intuition and Python practice.
    </p>
  </div>

  <div class="book-container">
    <img
      src="/books/raschka_book.jpg"
      alt="Book Cover Machine Learning with PyTorch and Scikit-Learn"
    />
    <p>
      The book "Machine Learning with PyTorch and Scikit-Learn" by Sebastian
      Raschka is similar in scope to the book by Aurélien Géron, but instead of
      using Keras and TensorFlow the author uses PyTorch. For the time being
      World4AI is going to provide notebooks written in PyTorch, so you will
      probably feel most comfortable with this book if you plan on using our
      notebooks.
    </p>
  </div>
  <div class="separator" />

  <h2>Online Courses</h2>
  <p>
    If you learn better through visual and audio explanations, online courses
    might be a great fit. Just as with our book recommendations, there is a lot
    of quality content for beginners and few resources for intermediary and
    advanced users.
  </p>
  <p>
    <a href="https://fast.ai" target="_blank" rel="noreferrer">fast.ai</a> is several things at the
    same time. It is a non-profit with the goal of democratizing deep learning, a
    deep learning library based on PyTorch, a book called 'Deep Learning for Coders
    With Fasai and Pytorch' and an online course that teaches you deep learning.
    The approach taken by fast.ai differs from most books and courses. They teach
    you how you can generate state of the art results first and go into the basic
    building blocks of neural networks second.
  </p>
  <p>
    The deep learning <a
      href="https://www.deeplearning.ai/program/deep-learning-specialization/"
      target="_blank" rel="noreferrer">coursera</a
    > course by Andrew Ng is probably the most famous and the most acknowledged course
    in deep learning. The course starts with the basic building blocks of deep learning
    and steadily moves towards the state of the art. The course is not free, but
    there is financial support available if you can not afford the lectures.
  </p>

  <p>
    The <a href="https://cds.nyu.edu/deep-learning/" target="_blank" rel="noreferrer"
      >NYU deep learning course</a
    > by Yann LeCun and Alfredo Canziani has become one of the most acknowledged
    courses over the last couple of years. The course is very in-depth and very engaging,
    but is probably not the best choice for absolute beginners. We recommend you
    take this course after you have had some exposure to an introductory course or
    book.
  </p>
  <div class="separator" />

  <h2>Deep Learning Practice</h2>
  <p>
    You can not learn deep learning without applying the algorithms with a deep
    learning framework yourself. Learning the theory is not going to be
    sufficient. If you have the data and an idea, then go for it and upload your
    project to GitHub. A popular alternative is to compete in machine learning
    challenges. The most popular plattform that host competitions is <a
      href="https://www.kaggle.com"
      target="_blank" rel="noreferrer">Kaggle</a
    >. There you can take part in discussions, upload datasets, create notebooks
    and take part in competitions. The plattform has a ranking system in the
    four mentioned categories and allows you to grow in the ranks and gain
    reputation and titles. Additionally many of the challenges give out monetary
    prices. Kaggle is essentially a social network, so if you want to get
    addicted to one, kaggle is a good choice for machine learning practitioners.
  </p>
  <div class="separator" />

  <h2>Keeping Up With Research</h2>
  <p>
    When you come from the outside of the deep learning research community, you
    do not have access to a personal network that you can utilize to find the
    most relevant research papers you should focus on. Fortunately you have
    access to twitter and can therefore follow companies that focus on deep
    learning research, established researchers and book authors to get access to
    your own individual network. Many of those sources will share interesting
    articles, tutorials, videos and the newest research. It will take some time
    to find which sources are the most reliable, but the benefits are enormous.
    We suggest that you create an account especially for the purpose of
    participating in the deep learning community and avoid other types of
    content. Otherwise you will experience the "dark side" of twitter.
  </p>
  <div class="separator" />
</Container>

<style>
  .book-container {
    display: flex;
    margin-bottom: 20px;
  }
  .book-container img {
    width: 200px;
    margin-right: 30px;
    margin-bottom: 10px;
    flex-grow: 1;
    height: fit-content;
  }

  .book-container p {
    margin: 0;
  }

  @media (max-width: 700px) {
    .book-container {
      flex-direction: column;
    }
  }
</style>
