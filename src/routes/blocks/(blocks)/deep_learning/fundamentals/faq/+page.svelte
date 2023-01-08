<script>
  import Container from "$lib/Container.svelte";
  import Alert from "$lib/Alert.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import GraphicsCard from "./_gpu_resources/GraphicsCard.svelte";
  import Blackboard from "./_education/Blackboard.svelte";

  // import images
  import CholetBook from "./_books/cholet_book.png";
  import GeronBook from "./_books/geron_book.jpeg";
  import RaschkaBook from "./_books/raschka_book.jpg";
</script>

<svelte:head>
  <title>FAQ Deep Learning - World4AI</title>
  <meta
    name="description"
    content="As a beginner you most like have a tons of questions regarding your setup, gpu resources and how to keep up to date. This section tries to answer those questions."
  />
</svelte:head>

<h1>FAQ</h1>
<div class="separator" />

<Container>
  <p>
    If you are an absolute beginner in deep learning, chances are you have a lot
    of questions. Questions like "what text editor is commonly used among deep
    learning practicioners?" or "what is the best deep learning library?", or
    "what books should I study?". We will not be able to answer all your
    questions, but we will try to provide you with enough answers so that you
    can start creating simple models and study on your own.
  </p>
  <div class="separator" />

  <h2>Code Editors</h2>
  <Alert type="info">
    What code editor should I use in my daily programming?
  </Alert>
  <p>
    <a href="https://code.visualstudio.com/" target="_blank" rel="noreferrer"
      >VS Code</a
    > (Visual Studio Code) by Microsoft is a very popular choice among researchers
    and practicioners. The editor has a lot of plugins and you can setup your environment
    withing a couple of minutes.
  </p>
  <p>
    Jupyter Notebooks with <a
      href="https://jupyter.org/"
      target="_blank"
      rel="noreferrer">JupyterLab</a
    > are a great way to test things out and to visualize your results. For larger
    projects it is not the preferred option. We will use notebooks extensively in
    this block, as notebookos are great for teaching purposes.
  </p>
  <Alert type="warning">
    We recommend Jupyter Notebooks for experimentation and VS Code for larger
    projects, but the choice is ultimately yours.
  </Alert>
  <div class="separator" />

  <h2>Frameworks</h2>
  <Alert type="info"
    >What programming languages and frameworks are suited for deep learning?</Alert
  >
  <p>
    Deep learning algorithms are usually implemented with specialized deep
    learning frameworks. These frameworks can run on the GPU to improve the
    speed of training and execution, have build in functionality that is
    commonly used in deep learning research and are battle tested. Implementing
    all those details by hand is usually a bad idea, unless you do it for
    learning purposes.
  </p>
  <p>
    Deep learning frameworks have a rich history. If we wrote this chapter two
    years ago the content would look different. Five years ago the contents
    would be different again. That says a lot about the speed with which the
    development of those frameworks progresses.
  </p>
  <p>
    All of the below mentioned frameworks use Python as the interface language
    between the programmer and the underlying execution. The frameworks
    themselves on the other hand are written in C++ and CUDA to optimize
    performance. There are frameworks that use other programming languages
    instead of Python like R, Julia or even JavaScript, but mostly research and
    education is done in Python so you are out of luck if you prefer any other
    programming language. Python is still king in the deep learning community.
  </p>

  <h3>TensorFlow and Keras</h3>
  <p>
    <a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"
      >TensorFlow</a
    >
    is Google's flagship deep learning framework. The framework was first released
    in the year 2015, which makes it the oldest in our list.
    <a href="https://keras.io/" target="_blank" rel="noreferrer">Keras</a> is a higher
    level api that sits on top of TensorFlow and provides a lot of convenience for
    developers by reducing a lot of overhead. Keras was actually developed independently
    by François Chollet in order to facilitate fast and easy deep learning development.
    Originally Keras supported multiple deep learning libraries (for example Theano),
    but became an integral part of TensorFlow 2 in 2019 and nowadays only supports
    TensorFlow. You can say that TensorFlow and Keras go hand in hand. Keras is an
    incredible great choice for beginners. It has a vibrant community and you will
    have no difficulty to find tutorials and books to learn from.
  </p>

  <h3>PyTorch</h3>
  <p>
    <a href="https://pytorch.org/" target="_blank" rel="noreferrer">PyTorch</a>
    is Meta's flagship deep learning framework, which was originally inspired by
    Torch, a Lua based deep learning library. In the years after the initial release
    in 2016, PyTorch has become a favorite among deep learning researchers and has
    overtaken TensorFlow.
  </p>

  <h3>JAX</h3>
  <p>
    <a href="https://github.com/google/jax" target="_blank" rel="noreferrer"
      >JAX</a
    > is the new kid on the block. Like Keras and TensorFlow, JAX is developed by
    Google. JAX has been gaining a lot of traction over the last couple of years.
  </p>

  <Alert type="warning">
    It is up to you which framework you choose. Especially when it comes to
    studying deep learning, all of the above frameworks are fine choices and are
    highly recommended. Keras and PyTorch have a much larger community and
    provide many more books and tutorials, but this might change within a couple
    of years. At the moment of writing most of our code examples are implemented
    in PyTorch.
  </Alert>
  <div class="separator" />

  <h2>GPU Resources</h2>
  <Alert type="info">Where can I get free or cheap GPU resources?</Alert>
  <SvgContainer maxWidth="500px">
    <GraphicsCard />
  </SvgContainer>
  <p>
    You require modern NVIDIA grapics cards for training and inference, but that
    option might not be available to everyone. If you are at the beginning of
    your carreer or live in a part of the world where it is unfeasable to afford
    your own computational resources, there are either cheap or free
    alternatives.
  </p>

  <h3>Free Cloud Resources</h3>
  <p>
    <a href="https://colab.research.google.com" target="_blank" rel="noreferrer"
      >Google Colab</a
    >
    (Colaboratory) provides you with free cloud compute in the form of Jupyter Notebooks.
    You can switch between CPU, GPU and TPU resources. If you use the platform extensively
    without a break they might switch you to a lower tier server or give you a time
    out, but considering the cost is zero, this is a great deal. Google also provides
    payed tiers with better GPUs and lower chance of getting disconnected. For learning
    purposes the free tier should be sufficient.
  </p>
  <p>
    <a href="https://www.kaggle.com" target="_blank" rel="noreferrer">
      Kaggle
    </a>

    is a plattform designed to run machine learning competitions. For that
    purpose they provide you weekly resources with at least 30 GPU hours and at
    least 30 TPU hours. Unlike Google Colab, Kaggle will not disconnect you, but
    when the hours run out you have to wait until the resources are reset. Kolab
    was acquired by Google, therefore the type of GPU or TPU you receive are
    comparable to those at Colab.
  </p>
  <p>
    If you are looking for alternatives free of Google,
    <a href="https://www.paperspace.com" target="_blank" rel="noreferrer"
      >Paperspace</a
    >
    might be it. Similar to Google Colab, Paperspace provides free cloud GPU notebooks
    for beginners and payed tiers for advanced users.
  </p>

  <h3>Paid Cloud Resources</h3>
  <p>
    There are tons of cloud providers that can hook you up with payed GPU
    instances. AWS by Amazon, Azure by Microsoft and Google Cloud are the top
    three players, but usually smaller alternatives like
    <a href="https://lambdalabs.com" target="_blank" rel="noreferrer">
      Lambda
    </a>
    or
    <a href="https://www.linode.com" target="_blank" rel="noreferrer">
      Linode
    </a> are much more buged friendly.
  </p>
  <div class="separator" />

  <h2>Education in Deep Learning</h2>
  <SvgContainer maxWidth={"600px"}>
    <Blackboard />
  </SvgContainer>
  <Alert type="info">
    What resources should I use for studying deep learning?
  </Alert>
  <p>
    We encourage you to consult many different sources on your deep learning
    journey. It usually requires more than one single resource to understand a
    topic. You should explore different books, videos, blog posts and research
    papers to get the full picture. Once you grasp the basics of deep learning,
    you still need to find a way to keep up to date with the newest research and
    this might not be the easiest task for a beginner. Below we attempt to list
    the most relevant literature for beginners and give hints on ways you can
    stay informed regarding the newest developments in deep learning.
  </p>

  <h3>Books</h3>
  <p>
    Books are a great gateway into the world of deep learning. Especially when
    it comes to beginner level books the choices are great.
  </p>

  <div
    class="flex flex-col justify-center md:justify-start items-center md:items-start md:flex-row gap-3 mb-4"
  >
    <img
      src={GeronBook}
      alt="Book Cover Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow"
      class="max-w-[200px]"
    />
    <p class="p-0 m-0">
      "Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow" by
      Aurélien Géron is a great choice, especially if you don't know where to
      start. If you have only one book to pick, this is probabliy the right
      pick. The book covers classical machine learning and goes on to discuss
      deep learning with Keras and TensorFlow in detail. While the book is very
      well suited for beginners, advanced topics like handling large amounts of
      data are covered as well.
    </p>
  </div>
  <div
    class="flex flex-col justify-center md:justify-start items-center md:items-start md:flex-row gap-3 mb-4"
  >
    <img
      src={CholetBook}
      alt="Book Cover DEEP LEARNING with Python"
      class="max-w-[200px]"
    />
    <p class="p-0 m-0">
      The book "DEEP LEARNING with Python" is written by the creator of Keras,
      so you know you get quality content. François Chollet covers exclusively
      deep learning with a high focus on intuition and Python practice.
    </p>
  </div>

  <div
    class="flex flex-col justify-center md:justify-start items-center md:items-start md:flex-row gap-3 mb-4"
  >
    <img
      src={RaschkaBook}
      alt="Book Cover Machine Learning with PyTorch and Scikit-Learn"
      class="max-w-[200px]"
    />
    <p class="p-0 m-0">
      The book "Machine Learning with PyTorch and Scikit-Learn" by Sebastian
      Raschka is similar in scope to the book by Aurélien Géron, but instead of
      using Keras and TensorFlow the author uses PyTorch. For the time being
      World4AI is going to provide notebooks written in PyTorch, so you will
      probably feel most comfortable with this book if you plan on using our
      notebooks.
    </p>
  </div>
  <h3>Online Courses</h3>
  <p>
    If you learn better through visual and audio explanations, online courses
    might be a great fit. Just as with our book recommendations, there is a lot
    of quality content for beginners and few resources for intermediary and
    advanced users.
  </p>
  <p>
    <a href="https://fast.ai" target="_blank" rel="noreferrer">fast.ai</a> is several
    things at the same time. It is a non-profit with the goal of democratizing deep
    learning, a deep learning library based on PyTorch, a book called 'Deep Learning
    for Coders With Fasai and Pytorch' and an online course that teaches you deep
    learning. The approach taken by fast.ai differs from most books and courses.
    They teach you how you can generate state of the art results first and go into
    the basic building blocks of neural networks second.
  </p>
  <p>
    The deep learning <a
      href="https://www.deeplearning.ai/program/deep-learning-specialization/"
      target="_blank"
      rel="noreferrer">coursera</a
    > course by Andrew Ng is probably the most famous and the most acknowledged course
    in deep learning. The course starts with the basic building blocks of deep learning
    and steadily moves towards the state of the art. The course is not free, but
    there is financial support available if you can not afford the lectures.
  </p>
  <p>
    The <a
      href="https://cds.nyu.edu/deep-learning/"
      target="_blank"
      rel="noreferrer">NYU deep learning course</a
    > by Yann LeCun and Alfredo Canziani has become one of the most acknowledged
    courses over the last couple of years. The course is very in-depth and very engaging,
    but is probably not the best choice for absolute beginners. We recommend you
    take this course after you have had some exposure to an introductory course or
    book.
  </p>
  <h3>Deep Learning Practice</h3>
  <p>
    You can not learn deep learning without applying the algorithms with a deep
    learning framework yourself. Learning the theory is not going to be
    sufficient. If you have the data and an idea, then go for it and upload your
    project to GitHub. A popular alternative is to compete in machine learning
    challenges. The most popular plattform that host competitions is <a
      href="https://www.kaggle.com"
      target="_blank"
      rel="noreferrer">Kaggle</a
    >. There you can take part in discussions, upload datasets, create notebooks
    and take part in competitions. The plattform has a ranking system in the
    four mentioned categories and allows you to grow in the ranks and gain
    reputation and titles. Additionally many of the challenges give out monetary
    prices. Kaggle is essentially a social network, so if you want to get
    addicted to one, kaggle is a good choice for machine learning practitioners.
  </p>
  <h3>Keeping Up With Research</h3>
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
