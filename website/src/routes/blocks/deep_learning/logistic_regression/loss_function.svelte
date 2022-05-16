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
  <p>
    This procedure is wasteful, because instead of sending one bit of
    information we send 2.
  </p>
  <p>
    The example below is even more inefficient. We sent 3 bits of information
    when we get heads and 2 bits when we get tails.
  </p>
  <Coin
    probHead={0.4}
    probTail={0.6}
    bitsHeads="1 0 1"
    bitsTails="0 1"
    showBits={true}
  />
  <p>
    The entropy of the probability distribution is just 0.97 bits. This is the
    amount of information we receive on average.
  </p>
  <Latex
    >-(0.4*\log_2(0.4) + 0.6*\log_2(0.6)) = {-(
      0.4 * Math.log2(0.4) +
      0.6 * Math.log2(0.6)
    ).toFixed(2)}</Latex
  >
  <p>
    Yet the average message length that we use to transimt the information is
    2.4 bits.
  </p>
  <Latex>0.4*3 + 0.6*2 = {(0.4 * 3 + 0.6 * 2).toFixed(2)}</Latex>
  <p>
    This inconsistency comes from certain assumtpions that we make in the
    example above. The encoding of heads with 3 bits would be optimal when the
    probability of each event would amount to 12.5%, as
    <Latex>\Big(\dfrac{1}{2}\Big)^3 = 0.125</Latex>
    and we had a coin with 8 sides. A similar argument can be made for 2 bits encoding
    of tails. We assume a probability distribution with four equally likely events,
    because
    <Latex>\Big(\dfrac{1}{2}\Big)^2 = 0.25</Latex>. Under these conditions we
    can say that the average message length contains information about the
    difference between two distributions, the actual distribution of the
    outcomes of coin tosses and the distribution that is assumed by the number
    of bits that is used to encode the message. In information theory the
    average message lengths is known as cross-entropy.
  </p>

  <p class="info">
    The cross-entropy is defined as the average message length.
  </p>
  <p>
    Given two distributions <Latex>p(x)</Latex> and <Latex>q(x)</Latex> we can calculate
    the cross-entropy <Latex>H(p, q)</Latex> by calculating the expected value with
    respect to the distribution <Latex>p</Latex> (the real distribution) of the information
    of distribution <Latex>q</Latex> (the wrong distribution).
  </p>
  <Latex
    >{String.raw`
    H(p, q) = - \mathbb{E}_p[\log q(x)] = - \sum_x p(x) \log q(x)
  `}</Latex
  >.
  <p>
    The cross-entropy is lowest, when <Latex>q(x)</Latex> equals <Latex
      >p(x)</Latex
    >. In that case the cross-entropy collapses to the entropy. Therefore the
    cross-entropy can not be lower than the entropy.
  </p>
  <p>
    In the below example the red distribution is <Latex>p(x)</Latex> and the yellow
    distribution is <Latex>q(x)</Latex>. When you move the slider to the right <Latex
      >q(x)</Latex
    > starts moving towards <Latex>p(x)</Latex> and you can observe that the cross-entropy
    gets lower and lower until its' minimal value is reached.
  </p>
  <CrossEntropy yTicks={[0, 0.1, 0.2, 0.3, 0.4]} />
  <p>
    Now it is time to come full circle and to relate the calculation of
    cross-entropy to our task. We set out to find a loss that is suited for
    classification tasks. Let us assume that we are dealing with a problem,
    where we have to classify an animal based on certain features in one of the
    five categories: cat, dog, pig, bear or monkey. The cross-entropy deals with
    probability distributions, so we need to put the label into a format that
    eqauls a probability distribution. For example if we deal with a sample that
    depicts a cat, the true probability distribution would be 100% for the
    category cat and 0% for all other categories. This distribution is put in a
    so called "one-hot" vector. A vector that contains a one for the relevant
    category and 0 otherwise. So that we have the following distributions.
  </p>
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
  <p>
    This distributions come from the labels, in the dataset. We therefore assume
    that the one hot vectors represent the correct distributions <Latex
      >p(x)</Latex
    >. Our neurons or neural networks also produce distributions that determine
    the category based on the labels.
  </p>
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
  <p>
    This is the estimated distribution and therefore the wrong distribution. We
    designate this distribution <Latex>q(x)</Latex>.
  </p>
  <p>
    Now we have everything to calculate the cross-entropy. The closer the one
    hot distribution and the distribution produced by the neural network get,
    the lower the cross-entropy gets. Because all the weight of the weight of
    the one hot vector is on just one event, the entropy corresponds to exactly
    0 and the cross-entropy could theoretically also reach 0. Our goal in a
    classification task is to minimize the cross-entropy to get the two
    distributions as close as possible.
  </p>
  <p>
    Below is an interactive example where the true label corresponds to the
    category cat. The estimated probabilities are far from the ground truth,
    which results in a relatively high cross-entropy. When you move the slider,
    the estimated probabilities start moving towards the ground truth, which
    pushes the cross-entropy down.
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
    <Latex
      >{String.raw`\sigma(b, \mathbf{w}) = \dfrac{1}{1 + e^{-(\mathbf{w^Tx}+b)}}`}</Latex
    >
    , which produces values between 0 and 1. The sigmoid function can be used to
    differentiate between 2 categories . The sigmoid function
    <Latex>{String.raw`\sigma(b, \mathbf{w})`}</Latex> produces the probability to
    belong to the first category (e.g. cat), therefore <Latex
      >{String.raw`1 - \sigma(b, \mathbf{w})`}</Latex
    > returns the probability to belong to the second category (e.g. dog). If we
    additionally define that the label <Latex>y</Latex> is 1 when the sample is a
    cat and 0 when the sample is a dog, the expression reduces the cross-entropy
    to the so called binary cross-entropy.
  </p>
  <Latex
    >{String.raw`
    H(p, \sigma) = -\Big[y \log ( \sigma(b, \mathbf{w})) + (1 - y) \log ( 1 - \sigma(b, \mathbf{w}))\Big]
  `}</Latex
  >.
  <p>
    When we shift the weights and the bias of the sigmoid function, we can move
    the probability to belong to a certain category. Our goal is therefore to
    find weigths that minimize the binary cross-entropy.
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
  <Latex
    >{String.raw`
    H(p, q) = -\dfrac{1}{n}\sum_i y^{(i)} \log ( \sigma(b, \mathbf{w})) + (1 - y^{(i)}) \log ( 1 - \sigma(b, \mathbf{w}))
  `}</Latex
  >
  <p>
    Just as with the linear regression, we will utilize gradient descent to find
    optimal weights. This is going to be covered in the next lecture.
  </p>
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
