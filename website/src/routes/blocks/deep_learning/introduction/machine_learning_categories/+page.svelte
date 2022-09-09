<script>
  import Highlight from "$lib/Highlight.svelte";
  import Table from "$lib/Table.svelte";
  import Plot from "$lib/Plot.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
  import CartPole from "$lib/reinforcement_learning/CartPole.svelte";

  let header = ["Location", "Size", "Price"];
  let data = [
    ["London", "100", 1_000_000],
    ["Berlin", "30", 80_000_000],
    ["...", "...", "..."],
  ];

  //data for unsupervised learning scatterplot
  let unsupervisedData = [
    [
      { x: 10, y: 15000 },
      { x: 20, y: 35000 },
      { x: 20, y: 25000 },
      { x: 25, y: 35000 },
      { x: 45, y: 50000 },
      { x: 15, y: 20000 },
      { x: 22, y: 25000 },
      { x: 33, y: 23000 },
      { x: 40, y: 37000 },
      { x: 27, y: 40000 },
    ],
    [
      { x: 110, y: 250000 },
      { x: 100, y: 235000 },
      { x: 90, y: 200000 },
      { x: 120, y: 700000 },
      { x: 135, y: 1000000 },
      { x: 140, y: 800000 },
      { x: 122, y: 325000 },
      { x: 133, y: 723000 },
      { x: 140, y: 457000 },
      { x: 127, y: 440000 },
    ],
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Machine Learning Categories</title>
  <meta
    name="description"
    content="Machine learning can be divided into different categories: supervised learning, unsupervised learning, self-supervised learning, semi-supervised learning and reinforcement learning."
  />
</svelte:head>

<Container>
  <h1>Categories of Machine Learning</h1>
</Container>
<div class="separator" />
<Container>
  <p>
    Machine learning is often divided into specific categories. Those
    classifications are usually done based on the level of human supervision
    that is required for the algorithm to work. Those classifications are
    ubiquitous nowadays, so knowing at least some basic terminology is a must.
  </p>
  <div class="separator" />

  <h2>Supervised Learning</h2>
  <p>
    As the name supervised learning suggests, there is a human supervisor who
    labels the input data with the corresponding correct output. The different
    inputs are called <Highlight>features</Highlight>, while the outputs are
    called <Highlight>labels</Highlight> or <Highlight>targets</Highlight>.
  </p>
  <p>
    Let us for example assume that we want to estimate the price of a house
    based on the location and the size of the house. In that case the location
    and the size are the features, while the price is the target.
  </p>
  <Table {header} {data} />
  <p>
    The two common tasks that we try to solve with supervised learning are <Highlight
      >regression</Highlight
    > and <Highlight>classification</Highlight>.
  </p>
  <p>
    In a classification task there is a finite number of classes that the
    machine learning algorithm needs to determine based on the features. The
    usual example that is brought up in the machine learning literature is a
    spam filter. Based on the header and content of the email the algorithm
    needs to decide whether the email is ham or spam.
  </p>

  <div class="light-blue">
    <SvgContainer maxWidth={"700px"}>
      <svg
        version="1.1"
        viewBox="0 0 500 250"
        xmlns="http://www.w3.org/2000/svg"
      >
        <g stroke="var(--text-color)">
          <g id="ham">
            <rect x="425" y="15" width="60" height="40" fill="none" />
            <path d="m425 15 30 20 30-20" fill="none" stroke-width="1px" />
          </g>
          <g id="filter">
            <rect
              x="160"
              y="45"
              width="120"
              height="120"
              fill="none"
              stroke-width="1px"
            />
            <g fill="none" stroke-width="0.5">
              <path d="m165 45v120" />
              <path d="m170 45v120" />
              <path d="m175 45v120" />
              <path d="m180 45v120" />
              <path d="m185 45v120" />
              <path d="m190 45v120" />
              <path d="m195 45v120" />
              <path d="m200 45v120" />
              <path d="m205 45v120" />
              <path d="m210 45v120" />
              <path d="m215 45v120" />
              <path d="m220 45v120" />
              <path d="m225 45v120" />
              <path d="m230 45v120" />
              <path d="m235 45v120" />
              <path d="m240 45v120" />
              <path d="m245 45v120" />
              <path d="m250 45v120" />
              <path d="m255 45v120" />
              <path d="m260 45v120" />
              <path d="m265 45v120" />
              <path d="m270 45v120" />
              <path d="m275 45v120" />
              <path d="m160 50h120" />
              <path d="m160 55h120" />
              <path d="m160 60h120" />
              <path d="m160 65h120" />
              <path d="m160 70h120" />
              <path d="m160 75h120" />
              <path d="m160 80h120" />
              <path d="m160 85h120" />
              <path d="m160 90h120" />
              <path d="m160 95h120" />
              <path d="m160 100h120" />
              <path d="m160 105h120" />
              <path d="m160 110h120" />
              <path d="m160 115h120" />
              <path d="m160 120h120" />
              <path d="m160 125h120" />
              <path d="m160 130h120" />
              <path d="m160 135h120" />
              <path d="m160 140h120" />
              <path d="m160 145h120" />
              <path d="m160 150h120" />
              <path d="m160 155h120" />
              <path d="m160 160h120" />
            </g>
          </g>
          <g id="spam">
            <rect
              x="433"
              y="169.5"
              width="40"
              height="46.495"
              ry="5.1661"
              fill="none"
            />
            <g fill="none" stroke-width=".91333px">
              <path d="m441 173.5v36.495" />
              <path d="m453 173.5v36.495" />
              <path d="m465 173.5v36.495" />
            </g>
            <rect
              x="433"
              y="165.5"
              width="40"
              height="4"
              ry="0"
              fill="#b8a0cd"
              fill-opacity=".054902"
              stroke-width=".78431"
            />
            <path d="m449 165.5v-4h8v4" fill="none" stroke-width=".8px" />
          </g>
          <g fill="none" stroke-dasharray="4,8">
            <path d="m70 105h80" />
            <path stroke="var(--main-color-2)" d="m285 100 135-62" />
            <path stroke="var(--main-color-1)" d="m285 115 140 77" />
          </g>
          <g id="mail">
            <rect
              x="9.3548"
              y="78.71"
              width="49.57"
              height="71.29"
              ry="1.371"
              fill="none"
            />
            <g fill="none">
              <path d="m13.28 85h41.832" />
              <path d="m13.28 90h41.832" />
              <path d="m13.28 95h41.832" />
              <path d="m13.28 100h41.72" />
              <path d="m13.28 105h41.72" />
              <path d="m13.28 110h41.72" />
              <path d="m13.28 115h41.72" />
            </g>
          </g>
        </g>
      </svg>
    </SvgContainer>
  </div>
  <p>
    In a regression task on the other hand the algorithm produces a continuous
    number. Predicting the price of the house based on the features of the house
    is a regression task.
  </p>

  <div class="yellow">
    <SvgContainer maxWidth={"700px"}>
      <svg
        version="1.1"
        viewBox="0 0 500 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="m110 50h275"
          fill="none"
          stroke="var(--text-color)"
          stroke-dasharray="4, 8"
        />
        <g fill="none" stroke="var(--text-color)">
          <path d="m35.535 85h38.93v-42.823h-38.93z" stroke-width=".7786px" />
          <path
            d="m31.642 46.07 23.358-31.144 23.358 31.144"
            stroke-width=".7786px"
          />
          <g stroke-width="1px">
            <path d="m50 83v-15h10v15z" />
            <path d="m40 50h5v5h-5z" />
            <path d="m70 50v5h-5v-5z" />
            <path d="m50 30h10v5h-10z" />
            <path d="m30 46 25-35 25 35" />
          </g>
        </g>
        <g fill="none" stroke="var(--text-color)">
          <path
            d="m16.624 175h81.753v-42.823h-81.753z"
            stroke-width="1.1283px"
          />
          <g stroke-width="1px">
            <path d="m50 173v-20s0-5 5-5 5 5 5 5v20z" />
            <path d="m20 135h5v5h-5z" />
            <path d="m30 135h5v5h-5z" />
            <path d="m30 145h5v5h-5z" />
            <path d="m20 145h5v5h-5z" />
            <path d="m20 155h5v5h-5z" />
            <path d="m30 155h5v5h-5z" />
            <path d="m85 145h5v5h-5z" />
            <path d="m75 145h5v5h-5z" />
            <path d="m75 155h5v5h-5z" />
            <path d="m85 155h5v5h-5z" />
            <path d="m5 130h105l-5-20h-95z" />
            <path d="m20 115v10h10v-10z" />
            <path d="m100 120v5h-20v-5z" />
          </g>
        </g>
        <path
          d="m110 150h270"
          fill="none"
          stroke="var(--text-color)"
          stroke-dasharray="3.96347, 7.92694"
          stroke-width=".99087"
        />
        <text
          x="384.08905"
          y="155.58189"
          fill="var(--text-color)"
          font-family="sans-serif"
          font-size="19.997px"
          stroke-width=".49992"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="384.08905" y="155.58189" stroke-width=".49992"
            >1,000,000$</tspan
          ></text
        >
        <text
          x="403.52597"
          y="55.515846"
          fill="var(--text-color)"
          font-family="sans-serif"
          font-size="19.997px"
          stroke-width=".49992"
          style="line-height:1.25"
          xml:space="preserve"
          ><tspan x="403.52597" y="55.515846" stroke-width=".49992"
            >200,000$</tspan
          ></text
        >
      </svg>
    </SvgContainer>
  </div>
  <p>
    In machine learning literature it is preferable to use the term label when
    we deal with classification tasks and the term target when we deal with
    regression tasks, but some authors might use those terms interchangebly.
  </p>
  <div class="separator" />

  <h2>Unsupervised Learning</h2>
  <p>
    In unsupervised learning the dataset contains only features and no labels.
    The overall task is to find some hidden structure in the data. We could for
    example use the labels in the house pricing dataset and cluster those houses
    according to the features of the houses. Similar houses would be allocated
    to the same cluster, while different houses should be in different clusters.
  </p>
  <p>
    In the below example we divide the houses into two categories based on size
    and price.
  </p>
  <Plot
    pointsData={unsupervisedData}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: 0,
      maxX: 150,
      minY: 0,
      maxY: 1000000,
      xLabel: "Size",
      yLabel: "Price",
      padding: { top: 20, right: 40, bottom: 40, left: 80 },
      radius: 5,
      colors: ["var(--main-color-1)", "var(--main-color-2)"],
      xTicks: [0, 50, 100, 150],
      yTicks: [0, 200000, 400000, 600000, 800000, 1000000],
    }}
  />
  <div class="separator" />

  <h2>Semi-supervised Learning</h2>
  <p>
    Labeling your dataset is costly, therefore companies and researchers try to
    get away with labeling as few samples as possible. In semi-supervised
    learning only a fraction of data has labels, while the rest is unlabeled.
    The labeled data is used to train the algorithm and to label the remaining
    data. After that step the whole dataset can be used for training.
  </p>
  <div class="separator" />

  <h2>Self-supervised Learning</h2>
  <p>
    Self-supervised learning can be seen as supervised learning, where the
    labels are not determined by a human supervisor, but are derived directly
    from the features of the data.
  </p>
  <p>Let us look for example at the sentence below.</p>
  <div class="sentence">
    <span class="word">What</span>
    <span class="word">is</span>
    <span class="word">your</span>
    <span class="word">name</span>
  </div>
  <p>
    We could design a natural language processing task by masking a part of the
    sentence.
  </p>
  <div class="sentence">
    <span class="word">What</span>
    <span class="word">is</span>
    <span class="word">your</span>
    <span class="word">...</span>
  </div>
  <p>
    The algorithm needs to learn to predict the masked word, which essentially
    becomes the label in our task. We feed the model with millions of such
    examples during the training process. Over time the model gets better and
    better at this task, which would indicate some knowledge about the structure
    of the english language. That pretrained model and by extension that
    knowledge about the english language can eventually be used in a supervised
    learning task like sentiment analysis, where you have only limited amount
    data.
  </p>
  <p>
    The process of learning a general task with a huge amount of data and using
    the same trained model which is recalibrated for a more specific task is
    called <Highlight>transfer learning</Highlight>. Self-supervised learning
    excels at transfer learning. Companies spend often millions of dollars to
    train such language models and in many cases they provide those to the
    general public, which can reuse them for their particular tasks.
  </p>
  <p>
    Self supervised learning is not limited to natural language processing.
    While the original ideas were developed for text, more recently
    self-supervised learning has also been applied successfully to computer
    vision.
  </p>
  <div class="separator" />

  <h2>Reinforcement Learning</h2>
  <p>
    Reinforcemnt learning deals with sequential decisions, where an agent
    interacts with the environment and receives rewards based on the preceding
    actions.
  </p>
  <CartPole showState={false} />
  <p>
    The cartpole is probably the most well known reinforcement learning task.
    The agent needs to learn to balance the pole by moving the cart left and
    right. Each single step the agent succeeds, the agent gets a reward of 1. If
    the pole gets below a certain angle or the cart moves outside the screen the
    agent fails and doesn't get any more rewards.
  </p>
  <div class="separator" />
</Container>

<style>
  .word {
    font-size: 26px;
    border: 1px solid var(--text-color);
    margin-right: 10px;
    padding: 0 5px;
    background-color: var(--main-color-3);
  }
</style>
