<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Coin from "./_entropy/Coin.svelte";
  import Entropy from "./_entropy/Entropy.svelte";
  import CrossEntropy from "./_entropy/CrossEntropy.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Path from "$lib/plt/Path.svelte";

  const notes = [
    "The properties that make the mean squared error a bad choice for classification tasks are discussed in the next section.",
    "A bit is a binary unit of of information. This unit is also referred to as a Shanon.",
    "In machine learning we often use nats for convenience (natural units), which use the base e instead of bits with the base of 2. This does not pose a problem, as we can simply convert from bits to nats by changing the base from 2 to e.",
  ];

  const references = [
    {
      author: "Shannon C. E.",
      title: "A Mathematical Theory of Communication",
      journal: "Bell System Technical Journal",
      year: "1948",
      pages: "379-423",
      volume: "27",
      issue: "3",
    },
  ];

  const bits = [1, 1, 0, 1, 1, 0, 0, 1];

  const informationData = [];
  function fillInformation() {
    for (let i = 0.001; i <= 1; i += 0.001) {
      let x = i;
      let y = -Math.log2(i);
      let data = { x, y };
      informationData.push(data);
    }
  }
  fillInformation();

  let theta = 0.1;
  // heads, heads, tails, heads, heads
  $: likelihood = theta * theta * (1 - theta) * theta * (1 - theta);

  // demonstrate log transform
  const transformData = [];

  let functionData = [];
  let logData = [];
  for (let i = -6; i < 6; i += 0.1) {
    let x = i;
    let y = x ** 2 + 0.5;
    functionData.push({ x, y });

    let log = Math.log(y);
    logData.push({ x, y: log });
  }
  transformData.push(functionData);
  transformData.push(logData);
</script>

<svelte:head>
  <title>Cross Entropy Loss - World4AI</title>
  <meta
    name="description"
    content="Cross-entropy, also called negative log likelihood, is the loss function that is used in classification tasks."
  />
</svelte:head>

<Container>
  <h1>Cross-Entropy Loss</h1>
  <div class="separator" />
  <p>
    The mean squared error loss tends to be problematic, when used as the loss
    function for classification tasks<InternalLink type="note" id={1} />. The
    loss that is usually used in classification tasks is called the <Highlight
      >cross-entropy</Highlight
    > (or the negative log likelihood loss).
  </p>

  <p>
    In 1948 Claude Shanon published an article called "A Mathematical Theory of
    Communication"<InternalLink type="reference" id={1} />. This paper
    introduced a theoretical foundation for a field that has become known as
    <Highlight>information theory</Highlight>.
  </p>
  <p>
    At first glance it might look like we are about to go on a tangent here,
    because information theory and the loss function for classification tasks
    should't have a lot in common. Yet the opposite is the case.
  </p>
  <Alert type="info">
    In order to understand the cross-entropy loss it is essential to understand
    information theory!
  </Alert>
  <p>
    We measure information using specific information units. The most common
    unit of information is the so called <Highlight>bit</Highlight><InternalLink
      id={2}
      type="note"
    />
    <InternalLink id={3} type="note" />, which takes a value of either 0 or 1.
    Below for example we use 8 bits to encode and send some information.
  </p>
  <div class="flex justify-center items-center gap-1">
    {#each bits as bit}
      <div class="border border-black py-1 px-2 md:px-3 bg-blue-100">
        {bit}
      </div>
    {/each}
  </div>
  <p>
    While we use 8 bits to send a message, we do not actually know how much of
    that information is useful. To get an intuition regarding that statement let
    us look at a simple toss coin example.
  </p>
  <p>
    Let us first imagine, that we are dealing with a fair coin, which means that
    the probability to get either heads or tails is exactly 50%.
  </p>
  <Coin probHead={0.5} probTail={0.5} />
  <p>
    To send a message regarding the outcome of the fair coin toss we need 1 bit.
    We could for example define heads as <span class="bit">1</span> and tails as
    <span class="bit">0</span>. The recepient of the message can remove the
    uncertainty regarding the coin toss outcome by simply looking at the value
    of the bit.
  </p>
  <p>
    But what if we deal with an unfair coin where heads comes up with a
    probability of 1.
  </p>
  <Coin probHead={1} probTail={0} />
  <p>
    We could still send 1 bit, but there would be no useful information
    contained in the message, because the recepient has no uncertainty regarding
    the outcome of the toss coin. Sending a bit in such a manner would be a
    waste of resources.
  </p>
  <p>Let's try to formalize the ideas we described above.</p>
  <Alert type="info">Information is inversely related to probability.</Alert>
  <p>
    We expect less likely events to provide more information than more likely
    events. In fact an event with a probability of 50% provides exactly 1 bit of
    information. Or to put it differently, one useful bit reduces uncertainty by
    exactly 2. Two bits of useful information reduce uncertainty by 4, three
    bits by 8 and so on.
  </p>
  <Alert type="info">
    We can convert probability <Latex>p</Latex> of an event <Latex>x</Latex> into
    bits of information <Latex>I</Latex> using the following equation.
    <div class="flex justify-center">
      <Latex>{String.raw`\Big(\dfrac{1}{2}\Big)^I = p(x)`}</Latex>
    </div>
    If the probability is 50%, the information content is exactly 1 bit. If the probability
    of an event is 25%, the uncertainty is divided by 4 when this event occurs and
    the information content is 2 bits.
  </Alert>
  <p />
  <p>
    We can use basic math to solve
    <Latex>{String.raw`\Big(\dfrac{1}{2}\Big)^I = p(x)`}</Latex>
    for information in bits <Latex>I</Latex>.
  </p>

  <Alert type="info">
    <Latex
      >{String.raw`
    \begin{aligned}
      &\Big(\frac{1}{2}\Big)^I = p(x) \\
      & 2^I = \frac{1}{p(x)} \\
      & I = \log_2\Big(\frac{1}{p(x)}\Big) \\
      & I = \log_2(1) - \log_2(p(x)) \\
      & I = 0 - \log_2(p(x)) \\
    \end{aligned} \\
    \boxed{I= -\log_2(p(x))}
  `}</Latex
    >
  </Alert>
  <p>
    We can plot the relationship between the probability <Latex>p</Latex> of an event
    <Latex>x</Latex> and the information measured in bits <Latex
      >-\log_2(p(x))</Latex
    >.
  </p>

  <Plot
    width={500}
    height={250}
    maxWidth={800}
    domain={[0, 1]}
    range={[0, 10]}
    padding={{ top: 10, right: 10, bottom: 40, left: 40 }}
  >
    <Ticks
      xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      yTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
      xOffset={-15}
      yOffset={15}
    />
    <XLabel text="Probability" fontSize={15} />
    <YLabel text="Number Of Bits" fontSize={15} />
    <Path data={informationData} />
  </Plot>
  <Alert type="info">
    The lower the probability of an event, the higher the information.
  </Alert>
  <p>
    Often we are not only interested in the amount of bits that is provided by a
    particular event. Instead we are interested in the expected value of
    information (the expected number of bits), that is contained in the whole
    probability distribution <Latex>p</Latex>. This measure is called <Highlight
      >entropy</Highlight
    >.
  </p>
  <Alert type="info">
    The entropy <Latex>H(p)</Latex> of the probability distrubution <Latex
      >p</Latex
    > is defined as the expected level of information or the expected number of bits.
    <div class="flex justify-center mt-1">
      <Latex
        >{String.raw`
    H(p) = -\sum_x p(x) * log_2(p(x))
  `}</Latex
      >
    </div>
  </Alert>
  <p>
    Below you can use an interactive example of a binomial distribution where
    you can change the probability of heads and tails. When you have a fair coin
    entropy amounts to exactly 1. When the probability starts getting uneven,
    entropy reduces until it reaches a value of 0.
  </p>
  <Entropy />
  <p>
    Intuitively speaking, the entropy is a measure of order of a probability
    distribution. Entropy is highest, when all the possible events have the same
    probability and entropy is 0 when one of the events has the probability 1
    while all other events have a probability of 0.
  </p>
  <p>
    Now let us return to the fair coin toss example. Using the equation and the
    example above, we know that the entropy is 1. We should therefore try and
    send the message with the result of the coin toss using 1 bit of
    information.
  </p>
  <Coin bitsHeads="1" bitsTails="0" showBits={true} />
  <p>
    In the example below on other hand we use an inefficient encoding. We always
    send 2 bits of information when we get heads and 2 bits when we get tails.
  </p>
  <Coin
    probHead={0.4}
    probTail={0.6}
    bitsHeads="1 0"
    bitsTails="0 1"
    showBits={true}
  />
  <p>The entropy of the probability distribution is just 0.97 bits.</p>
  <div class="flex justify-center">
    <Latex
      >-(0.4*\log_2(0.4) + 0.6*\log_2(0.6)) = {-(
        0.4 * Math.log2(0.4) +
        0.6 * Math.log2(0.6)
      ).toFixed(2)}</Latex
    >
  </div>
  <p>
    Yet the average message length, also known as <Highlight
      >cross-entropy</Highlight
    > is 2 bits.
  </p>
  <div class="flex justify-center">
    <Latex>0.4*2 + 0.6*2 = {(0.4 * 2 + 0.6 * 2).toFixed(2)}</Latex>
  </div>
  <p>
    By using 2 bits to encode the message, we implicitly assume a different
    probability distribution, than the one that produced the coin toss. Remember
    that 2 bits would for example correspond to a distribution with 4 likely
    events, each occuring with a probability of 25%. In a way we can say, that
    the cross-entropy allows us to measure the difference between two
    distributions. Only when the distribution that produced the event and the
    distribution we use to encode the message are identical, does the
    cross-entropy reach its minimum value. In that case the cross-entropy and
    the entropy are identical.
  </p>
  <Alert type="info">
    The cross-entropy is defined as the average message length. Given two
    distributions <Latex>p(x)</Latex> and <Latex>q(x)</Latex> we can calculate the
    cross-entropy <Latex>H(p, q)</Latex>.
    <div class="flex justify-center mt-1">
      <Latex
        >{String.raw`
    H(p, q) = - \mathbb{E}_p[\log q(x)] = - \sum_x p(x) \log q(x)
  `}</Latex
      >.
    </div>
  </Alert>
  <p>
    In the below example the red distribution is <Latex>p(x)</Latex> and the yellow
    distribution is <Latex>q(x)</Latex>. When you move the slider to the right <Latex
      >q(x)</Latex
    > starts moving towards <Latex>p(x)</Latex> and you can observe that the cross-entropy
    gets lower and lower until its' minimal value is reached. In that case the two
    distributions are identical and the cross-entropy is equal to the entropy.
  </p>
  <CrossEntropy yTicks={[0, 0.1, 0.2, 0.3, 0.4]} />
  <p>
    Now it is time to come full circle and to relate the calculation of the
    cross-entropy to our initial task: find a loss function that is suited for
    classification tasks.
  </p>
  <p>
    Let us assume that we are dealing with a problem, where we have to classify
    an animal based on certain features in one of the five categories: cat, dog,
    pig, bear or monkey. The cross-entropy deals with probability distributions,
    so we need to put the label into a format that equals a probability
    distribution. For example if we deal with a sample that depicts a cat, the
    true probability distribution would be 100% for the category cat and 0% for
    all other categories. This distribution is put in a so called "one-hot"
    vector. A vector that contains a one for the relevant category and 0
    otherwise. So that we have the following distributions, <Latex>p(x)</Latex>.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
  \text{cat} = 
  \begin{bmatrix}
  1 \\ 
  0 \\ 
  0 \\ 
  0 \\ 
  0 \\ 
  \end{bmatrix}
  \text{dog} = 
  \begin{bmatrix}
  0 \\ 
  1 \\ 
  0 \\ 
  0 \\ 
  0 \\ 
  \end{bmatrix}
  \text{pig} = 
  \begin{bmatrix}
  0 \\ 
  0 \\ 
  1 \\ 
  0 \\ 
  0 \\ 
  \end{bmatrix}
  \text{bear} = 
  \begin{bmatrix}
  0 \\ 
  0 \\ 
  0 \\ 
  1 \\ 
  0 \\ 
  \end{bmatrix}
  \text{monkey} = 
  \begin{bmatrix}
  0 \\ 
  0 \\ 
  0 \\ 
  0 \\ 
  1 \\ 
  \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    The distributions that are produced by the sigmoid or the softmax functions
    on the other hand are just estimations. We designate this distribution <Latex
      >q(x)</Latex
    >.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`

  \begin{bmatrix}
  0.05 \\ 
  0.4 \\ 
  0.05 \\ 
  0.4 \\ 
  0.1 \\ 
  \end{bmatrix}
    `}</Latex
    >
  </div>
  <p>
    Now we have everything to calculate the cross-entropy. The closer the one
    hot distribution and the distribution produced by the logistic regression or
    the neural network get, the lower the cross-entropy gets. Because all the
    weight of the one hot vector is on just one event, the entropy corresponds
    to exactly 0, which means the cross-entropy could theoretically also reach
    0. Our goal in a classification task is to minimize the cross-entropy to get
    the two distributions as close as possible.
  </p>
  <p>
    Below is an interactive example where the true label corresponds to the
    category cat. The estimated probabilities are far from the ground truth,
    which results in a relatively high cross-entropy. When you move the slider,
    the estimated probabilities start moving towards the ground truth, which
    pushes the cross-entropy down, until it reaches a value of 0.
  </p>
  <CrossEntropy
    points1={[
      { event: "Cat", percentage: 1 },
      { event: "Dog", percentage: 0 },
      { event: "Pig", percentage: 0 },
      { event: "Bear", percentage: 0 },
      { event: "Monkey", percentage: 0 },
    ]}
    startingPoints={[
      { event: "x1", percentage: 0.05 },
      { event: "x2", percentage: 0.4 },
      { event: "x3", percentage: 0.05 },
      { event: "x4", percentage: 0.4 },
      { event: "x5", percentage: 0.1 },
    ]}
  />
  <p>
    In logistic regression we utilize the sigmoid activation function
    <Latex>{String.raw`\hat{y} = \dfrac{1}{1 + e^{-(\mathbf{w^Tx}+b)}}`}</Latex>
    , which produces values between 0 and 1. The sigmoid function can be used to
    differentiate between 2 categories . The sigmoid function produces the probability
    <Latex>{String.raw`\hat{y}`}</Latex> to belong to the first category (e.g. cat),
    therefore <Latex>{String.raw`1 - \hat{y}`}</Latex> returns the probability to
    belong to the second category (e.g. dog). If we additionally define that the
    label <Latex>y</Latex> is 1 when the sample is a cat and 0 when the sample is
    a dog, the expression reduces the cross-entropy to the so called <Highlight
      >binary cross-entropy</Highlight
    >.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    H(p, q) = -\Big[y \log (\hat{y}) + (1 - y) \log ( 1 - \hat{y})\Big]
  `}</Latex
    >
  </div>
  <Alert type="info">
    When we are dealing with a classification problem we use the cross-entropy
    as the loss function. We use the binary cross-entropy when we have just 2
    categories.
  </Alert>
  <p>
    When we shift the weights and the bias of the sigmoid function, we can move
    the probability to belong to a certain category closer to the ground truth
    in order to reduce the cross-entropy. In the next section we will
    demonstrate how we can we can utilize gradient descent for that purpose. For
    now you can play with the interactive example below.
  </p>
  <CrossEntropy
    maxWidth={"400px"}
    points1={[
      { event: "Cat", percentage: 1 },
      { event: "Dog", percentage: 0 },
    ]}
    startingPoints={[
      { event: "x1", percentage: 0.25 },
      { event: "x2", percentage: 0.75 },
    ]}
  />
  <p>
    In practice we always deal with with a dataset, therefore the cross-entropy
    loss that we are going to optimize is going to be the average over the whole
    dataset.
  </p>
  <Alert type="info">
    <Latex
      >{String.raw`
    H(p, q) = -\dfrac{1}{n}\sum_i y^{(i)} \log (\hat{y}^{(i)}) + (1 - y^{(i)}) \log ( 1 - \hat{y}^{(i)})
  `}</Latex
    >
  </Alert>
</Container>

<Footer {notes} {references} />
