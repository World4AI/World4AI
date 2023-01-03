<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";

  // table
  import Table from "$lib/base/table/Table.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";

  import Function from "../_ml_definition/Function.svelte";
  import CartPole from "$lib/reinforcement_learning/CartPole.svelte";

  let references = [
    {
      title: "Some Studies in Machine Learning Using the Game of Checkers",
      author: "Samuel A.L",
      journal: "IBM Journal of Research and Development",
      volume: 44,
      pages: "206-226",
      year: 1959,
    },
  ];

  let spamHeader = ["ID", "Address", "Subject", "Expected Output"];
  let spamData = [
    [1, "nigerian.prince@ng-gov.ng", "Help Me", "spam"],
    [2, "marta.smith@gmail.com", "Tax Report", "ham"],
    ["...", "...", "...", "..."],
    [1000, "no-reply@info.o2.com", "New Contract", "ham"],
  ];

  let header = ["Location", "Size", "Price"];
  let data = [
    ["London", "100", 1_000_000],
    ["Berlin", "30", 80_000_000],
    ["...", "...", "..."],
  ];

  //data for unsupervised learning scatterplot
  const category1 = [
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
  ];
  const category2 = [
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
  ];
</script>

<svelte:head>
  <title>Machine Learning - World4AI</title>
  <meta
    name="description"
    content="Machine learning is a programming paradigm, where the logic of a program is learned from data."
  />
</svelte:head>

<h1>Machine Learning</h1>
<div class="separator" />
<Container>
  <h2>ML Definition</h2>
  <p>
    Let's start this section at the very beginning and define the term <Highlight
      >machine mearning</Highlight
    >.
  </p>
  <Alert type="info">
    "Machine learning is the field of study that gives computers the ability to
    learn without being explicitly programmed<InternalLink
      type="reference"
      id={1}
    />".
  </Alert>
  <p>
    The above definition is inspired by Arthur Samuel, one of the pioneers in
    the area of artificial intelligence, who coined the term machine learning.
    While this definition is commonly used, it is not the one that we will rely
    on. Throughout the deep learning block we will rely on a much more simple,
    more programming oriented definition of machine learning.
  </p>
  <Alert type="info">Machine learning is a programming paradigm.</Alert>
  <p>
    Let's take some time and figure out what that definition actually means.
  </p>
  <p>
    In simplified terms we can say, that the task of the programmer is to write
    a function, that can generate desired outputs based on the inputs to the
    function.
  </p>
  <Function />
  <p>
    For example a programmer might be assigned the task to write a spam filter,
    where the function would classify the email as spam or ham based on the
    contents of the email, the email address, the email subject and some
    additional metadata. It does not matter whether the programmer uses a
    traditional programming paradigm or machine learning, the result of the task
    is essentially the same: a function that takes those inputs and produces
    email classification as the output. The big difference between the classical
    and the machine learning progamming paradigm is the way that this function
    is derived.
  </p>
  <p>
    When programmers apply a traditional programming paradigm to create a spam
    filter, they will study the problem at hand and look at the inputs of the
    function. They could for example recognize that the words <em
      >money, rich and quick</em
    > are common in spam emails and write the first draft of the the function using
    a programming language like JavaScript or C++. If the output of the function
    corresponds to the expectations of the programmers, the job is done. If not,
    the programmers would keep improving the code of the function until the outputs
    of the function are satisfactory. For example the programmers might be satisfied,
    once the produced function is able to classify spam emails with an accuracy of
    95%.
  </p>
  <p>
    The machine learning paradigm is a different approach. While both paradigms
    produce a function, in machine learing we commonly tend to use the word <Highlight
      >model</Highlight
    > instead of function. The programmer still needs to write some parameters of
    the model explicitly, but the logic of the function is configured in an automated
    procedure called <Highlight>model training</Highlight>. For that purpose the
    programmer needs to have access to a <Highlight>dataset</Highlight>that
    contains the inputs to the function and the correct desired outputs.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each spamHeader as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each spamData as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>
    The model takes in the inputs from the dataset (address and subject) and
    predicts the outputs (spam or ham). Using the difference between the actual
    outputs and the outputs produced by the model, the logic of the model is
    adjusted automatically. That procedure keeps repeating in a loop until some
    metric is met. As the performance of the model is generally expected to
    improve over time, we also tend to call this procedure <Highlight
      >learning</Highlight
    >.
  </p>
  <Alert type="info">
    In classical programming and machine learning we try to solve a problem by
    generating computer functions. In classical programming the programmer codes
    the logic of that function explicitly. In machine learning the programmer
    provides the dataset and chooses the algorithm and model parameters that are
    used to learn the function.
  </Alert>
  <p>
    One final question remains: "When do we use machine learning and when do we
    use classical programming?". Machine learning is usually used when the
    complexity of the program would get out of hand if we implemented the logic
    manually. A program that is able to recognize digits is almost impossible to
    implement by hand. How would you for example implement a program that is
    able to differentiate between an 8 and a 9? This is an especially hard
    problem when the location of the numbers is scattered and not centered in
    the middle of an image. The same problem can be solved relatively
    straightforward using neural networks, provided we have the necessary data.
  </p>
  <div class="separator" />

  <h2>ML Categories</h2>
  <p>
    Machine learning is often divided into specific categories. Those
    classifications are ubiquitous nowadays, so knowing at least some basic
    terminology is a must.
  </p>

  <h3>Supervised Learning</h3>
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
  <Table>
    <TableHead>
      <Row>
        {#each header as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each data as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
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

  <div class="bg-slate-100">
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

  <div class="bg-slate-100">
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

  <h3>Unsupervised Learning</h3>
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
    maxWidth={600}
    domain={[0, 150]}
    range={[0, 1000000]}
    padding={{ top: 40, right: 10, left: 72, bottom: 40 }}
  >
    <Ticks
      xTicks={[0, 50, 100, 150]}
      xOffset={-15}
      yOffset={35}
      yTicks={[0, 250000, 500000, 750000, 1000000]}
    />
    <XLabel text="Size" fontSize={20} />
    <YLabel text="Price" fontSize={20} />
    <Circle data={category1} />
    <Circle data={category2} color={"var(--main-color-2)"} />
  </Plot>

  <h3>Semi-supervised Learning</h3>
  <p>
    Labeling your dataset is costly, therefore companies and researchers try to
    get away with labeling as few samples as possible. In semi-supervised
    learning only a fraction of data has labels, while the rest is unlabeled.
    The labeled data is used to train the algorithm and to label the remaining
    data. After that step the whole dataset can be used for training.
  </p>

  <h3>Self-supervised Learning</h3>
  <p>
    Self-supervised learning can be seen as supervised learning, where the
    labels are not determined by a human supervisor, but are derived directly
    from the features of the data.
  </p>
  <p>Let us look for example at the sentence below.</p>
  <div class="flex gap-1 justify-center">
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">What</span>
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">is</span>
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">your</span>
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">name</span>
  </div>
  <p>
    We could design a natural language processing task by masking a part of the
    sentence.
  </p>
  <div class="flex gap-1 justify-center">
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">What</span>
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">is</span>
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">your</span>
    <span class="bg-w4ai-yellow p-1 text-xl border border-black">...</span>
  </div>
  <p>
    The algorithm needs to learn to predict the masked word, which essentially
    becomes the label in our task. We feed the model with millions of such
    examples during the training process. Over time the model gets better and
    better at this task, which would indicate some knowledge about the structure
    of the english language. That model and by extension that language can
    eventually be used in a supervised learning task like sentiment analysis,
    where you have only limited amount data.
  </p>
  <p>
    Self supervised learning is not limited to natural language processing.
    While the original ideas were developed for text, more recently
    self-supervised learning has also been applied successfully to computer
    vision.
  </p>

  <h3>Reinforcement Learning</h3>
  <p>
    Reinforcemnt learning deals with sequential decisions, where an agent
    interacts with the environment and receives rewards based on its actions.
  </p>
  <CartPole showState={false} />
  <p>
    The cartpole is probably the most well known reinforcement learning task.
    The agent needs to learn to balance the pole by moving the cart left or
    right. Each single step the agent succeeds, the agent gets a reward of 1. If
    the pole gets below a certain angle or the cart moves outside the screen the
    agent fails and doesn't get any more rewards.
  </p>
</Container>
<Footer {references} />
