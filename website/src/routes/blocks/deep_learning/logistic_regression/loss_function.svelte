<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Coin from "./_entropy/Coin.svelte";
  import Plot from "$lib/Plot.svelte";
  import Entropy from "./_entropy/Entropy.svelte";
  import CrossEntropy from "./_entropy/CrossEntropy.svelte";

  const notes = [
    "The properties that make the mean squared error a bad choice for classification tasks are discussed down below after the introduction of cross entorpy",
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
</script>

<Container>
  <h1>Loss Function</h1>
  <div class="separator" />
  <p>
    The mean squared error tends to be problematic, when used as the loss
    function for classification tasks<InternalLink type="note" id={1} />. The
    loss that is usually used in classification tasks is either called
    cross-entropy or negative log likelihood. Both names are used in the
    literature and the standard machine learning libraries and while the
    calculation is the same, the motivation (and therefore the derivation) is
    different. We will introduce both explanations in order to get a better
    understanding and intuition for this type of loss.
  </p>
  <div class="separator" />

  <h2>Entropy and Cross-Entropy</h2>
  <p>
    In 1948 Claude Shanon published an article called "A Mathematical Theory of
    Communication"<InternalLink type="reference" id={1} />. This paper
    introduced a theoretical foundation for a field that has become known as
    information theory.
  </p>
  <p>
    At first glance it might look like we are about to go on a tangent here,
    because information theory and the loss function for classification tasks
    should't have a lot in common. Yet the opposite is the case.
  </p>
  <p class="info">
    To understand cross-entropy loss is to understand information theory!
  </p>
  <p>
    We measure information using specific information units. The most common
    unit of information is the so called bit<InternalLink id={2} type="note" />
    <InternalLink id={3} type="note" />. A bit takes a value of either 0 or 1.
    Below for example we use 8 bits to encode and send some information.
  </p>
  <p class="bits">
    <span class="bit">1</span><span class="bit">1</span><span class="bit"
      >0</span
    ><span class="bit">1</span><span class="bit">1</span><span class="bit"
      >0</span
    ><span class="bit">0</span><span class="bit">1</span>
  </p>
  <p>
    But there is a difference between using 8 bits to send a message and the
    amout of information that is actually contained in the message. To get an
    intuition regarding that statement let us look at a toss of a coin as an
    example.
  </p>
  <Coin probHead={0.5} probTail={0.5} />
  <p>
    To send a message regarding the outcome of the fair coin toss we need 1 bit.
    We could for example define heads as <span class="bit">1</span> and tails as
    <span class="bit">0</span>.
  </p>
  <p>
    But what if we deal with an unfair coin where heads comes up with a
    probability of 1.
  </p>
  <Coin probHead={1} probTail={0} />
  <p>
    We could still send 1 bit, but there would be no useful information
    contained in the message. As the probability is always 100%, there would be
    0 information provided.
  </p>
  <p>
    Intuitively speaking information is the level of surprise. Information is
    inversely related to probability, therefore we expect less likely events to
    provide more information than more likely events. In fact an event with a
    probability of 50% provides exactly 1 bit of information. Or to put it
    differently one bit reduces uncertainty by exactly 2. 2 bits of information
    reduce the uncertainty by 4, 3 bits by 8 and so on.
  </p>
  <p>
    We can generalize this idea using the equation: <Latex
      >{String.raw`(\frac{1}{2})^I = p(x)`}</Latex
    >, where <Latex>p</Latex> is a probability of an event <Latex>x</Latex> and <Latex
      >I</Latex
    > is the information in bits. If the probability is 50%, the information content
    is exactly 1 bit. If the probability of an event is 25%, the uncertainty is divided
    by 4 when this event occurs and the information content is 2 bits.
  </p>
  <p>
    We can use basic math to solve
    <Latex>{String.raw`(\frac{1}{2})^I = p(x)`}</Latex>
    for information in bits.
  </p>
  <div class="latex-box">
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
  </div>
  <p>
    The plot below shows this relationship between the probability and the
    information measured in bits. The lower the probability the higher the
    surprise and the information in bits when the event occurs.
  </p>
  <Plot
    pathsData={informationData}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: 0,
      maxX: 1,
      minY: 0,
      maxY: 10,
      xLabel: "Probability",
      yLabel: "Number of Bits",
      padding: { top: 20, right: 40, bottom: 40, left: 40 },
      xTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      yTicks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }}
  />

  <p>
    Often we are not only interested in the amount of bits that is provided by a
    single event, but in the expected value of information (the expected number
    of bits) that is contained in the whole probability distribution. This
    measure is called entropy.
  </p>
  <p class="info">
    The entropy <Latex>H(p)</Latex> of the probability distrubution <Latex
      >p</Latex
    > is defined as the expected level of information or the expected number of bits.
  </p>
  <Latex
    >{String.raw`
    H(p) = -\sum_x p(x) * log_2(p(x))
  `}</Latex
  >
  <p>
    Intuitively the information entropy is a measure of order of a probability
    distribution. Entropy is highest, when all the possible events have the same
    probability and entropy is 0 when one of the events has the probability 1
    while all other events have a probability of 0. Below you can use an
    interactive example of a binomial distribution where you can change the
    probability of heads and tails. When you have a fair coin entropy amounts to
    exactly 1. When the probability starts getting uneven, entropy reduces until
    it reaches a value of 0.
  </p>
  <Entropy />

  <p>
    Now let us return to the fair coin toss example. Using the equation and the
    example above, we know that the entropy is 1. We should therefore try and
    send the message regarding the result of the coin toss using 1 bit of
    information using 1 for heads and 0 for tails. However in the example below
    we use two bits of information to send the message 1 0 for heads and 0 1 for
    tails.
  </p>
  <Coin bitsHeads="1 0" bitsTails="0 1" showBits={true} />

  <Coin bitsHeads="1 0 1" bitsTails="0 1" showBits={true} />
  <p class="info">The cross-entropy is defined as the average message length</p>
  <Latex
    >{String.raw`
    H(p, q) = - \mathbb{E}_p[\log q(x)] = - \sum_x p(x) \log q(x)
  `}</Latex
  >
  <CrossEntropy yTicks={[0, 0.1, 0.2, 0.3, 0.4]} />
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
  <div class="separator" />

  <h2>Negative Log Likelihood</h2>
  <div class="separator" />

  <h2>Cross-Entropy and Mean Squared Error</h2>
  <div class="separator" />
</Container>

<Footer {notes} {references} />

<style>
  .bits {
    width: fit-content;
    background-color: var(--main-color-3);
    padding: 2px 15px;
  }
  .bit {
    color: black;
    border: 1px solid black;
    padding: 0 2px;
    margin-right: 5px;
  }

  .latex-box {
    padding: 10px 19px;
    background-color: var(--main-color-3);
    width: fit-content;
  }
</style>
