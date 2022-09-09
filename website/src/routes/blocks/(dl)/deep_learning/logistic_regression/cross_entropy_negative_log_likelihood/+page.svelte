<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Coin from "../_entropy/Coin.svelte";
  import Plot from "$lib/Plot.svelte";
  import Entropy from "../_entropy/Entropy.svelte";
  import CrossEntropy from "../_entropy/CrossEntropy.svelte";
  import Slider from "$lib/Slider.svelte";
  import Highlight from "$lib/Highlight.svelte";

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
  <title
    >World4AI | Deep Learning | Cross Entropy and Negative Log Likelihood</title
  >
  <meta
    name="description"
    content="Cross-entropy, also called negative log likelihood, is the loss function that is used in classification tasks."
  />
</svelte:head>

<Container>
  <h1>Cross-Entropy and Negative Log Likelihood</h1>
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
  <p>
    There is a second approach to cover the loss function that is used in
    classification tasks. This is done by following the path of maximum
    likelihood estimation.
  </p>
  <p>
    Let us start with a small revision of probability theory and statistics and
    answer the following question: <span class="light-blue"
      >"What is the difference between probability and likelihood?"<span
        >.
      </span></span
    >
    We will demonstrate the difference using our trusted coin toss example.
  </p>
  <p>
    The result of the coin toss <Latex>{String.raw`y^{(i)}`}</Latex> has a Bernoulli
    distribution, <Latex>{String.raw`y^{(i)} \sim Ber(\theta)`}</Latex>. With a
    probability of <Latex>\theta</Latex> we toss heads and with probability of <Latex
      >{String.raw`1-\theta`}</Latex
    > we toss tails. If we define the results of heads as 1 and the results of tails
    as 0, we can define the parameter
    <Latex>\theta</Latex> as <Latex>{String.raw`\Pr(y^{(i)} = 1)`}</Latex> and <Latex
      >1 - \theta</Latex
    > as <Latex>{String.raw`\Pr(y^{(i)} = 0)`}</Latex>. The PMF (probability
    mass function) is therefore <Latex
      >{String.raw`p(y^{(i)} | \theta) = \theta^{y^{(i)}} (1 - \theta)^{1 - y^{(i)}}`}</Latex
    >. To understand the PMF imagine the following. In case we need to calculate
    the probability for heads, we replace <Latex>{String.raw`y^{(i)}`}</Latex> by
    1 and end up with just <Latex>\theta</Latex>. On the other hand we can use
    the PMF to calculate the probability for tails by replacing <Latex
      >{String.raw`y^{(i)}`}</Latex
    > by 0. In that case we end up with <Latex>1-\theta</Latex>. Lets clarify
    those definitions with a simple example.
  </p>
  <Coin probHead={0.3} probTail={0.7} />
  <p>
    We are faced with an unfair coin, where <Latex>\theta</Latex> is 0.3 and <Latex
      >1 - \theta</Latex
    > is 0.7. The PMF is <Latex>{String.raw`p(y^{(i)}|\theta)`}</Latex> resolves
    to <Latex>{String.raw`p(1 | 0.3) = 0.3`}</Latex> and <Latex
      >{String.raw`p(0 | 0.3) = 0.7`}</Latex
    > respectively. Often we are interested in the probability of a particular sequence
    of coin tosses,
    <Latex>{String.raw`p(y^{(1)}, y^{(2)}, \cdots, y^{(n)}|\theta)`}</Latex>
    . All coin tosses are independently distributed, it does not matter what coin
    tosses came before. That means we can express the probability of the sequence
    as the product of probabilities.
  </p>
  <p>
    <Latex
      >{String.raw`
 \begin{aligned}
 & p(y^{(1)}, y^{(2)}, \cdots, y^{(n)}|\theta) = \\
& = p(y^{(1)} | \theta) * p(y^{(2)}|\theta)  * \cdots * p(y^{(n)}|\theta) \\
& = \prod_{i=1}^n p(y^{(i)}|\theta)
  \end{aligned}
 `}</Latex
    >
  </p>
  <p>
    For example: "What is the probability to get the sequence of Heads, Heads,
    Tails, Heads, Tails, when <Latex>\theta</Latex> equals 0.3? Or expressed mathematically:
    <Latex>{String.raw`p(1, 1, 0, 1, 0 | 0.3)`}</Latex>.
  </p>
  <Latex
    >{String.raw`
 \begin{aligned}
 & p(y^{(1)}, y^{(2)},  y^{(3)},y^{(4)},y^{(5)} | \theta)= \\
 & = p(1, 1, 0, 1, 0 | 0.3) \\ 
& = 0.3 * 0.3 * 0.7 * 0.3 * 0.7 \\
& = 0.01323 \\  
& = 1.323 \%
  \end{aligned}
 `}</Latex
  >
  <p>
    When we are dealing with probabilities, the function<Latex
      >{String.raw`
p(y^{(1)}, y^{(2)}, \cdots, y^{(n)}|\theta) 
      `}</Latex
    > has a fixed parameter <Latex>\theta</Latex> and expects observations (e.g.
    coin tosses) as inputs to the function. The PMF <Latex>p</Latex> returns the
    corresponding probability.
  </p>
  <p>
    Now imagine we do not know the parameter <Latex>\theta</Latex>, instead we
    toss a coin many times to gain access to observations <Latex
      >{String.raw`y^{(1)}, y^{(2)}, \cdots, y^{(n)}`}</Latex
    >. The likelihood function
    <Latex
      >{String.raw`\mathcal{L}(\mathbf{\theta} | y^{(1)}, y^{(1)},\cdots y^{(n)})`}</Latex
    > returns the likelihood that the parameter <Latex>\theta</Latex> is the parameter
    of the PMF, given that we collected the samples <Latex
      >{String.raw`y^{(1)}, y^{(2)}, \cdots, y^{(n)}`}</Latex
    >.
  </p>
  <p>
    There is a straightforward relatinoship between the likelihood <Latex
      >{String.raw`\mathcal{L}`}</Latex
    > and the probability <Latex>p</Latex>. Given the same parameter values <Latex
      >{String.raw`\theta`}</Latex
    > and the same observations
    <Latex
      >{String.raw`
y^{(1)}, y^{(1)},\cdots y^{(n)}
        `}</Latex
    > the probability and the likelihood are equal.
  </p>

  <Latex
    >{String.raw`
\mathcal{L}(\mathbf{\theta} | y^{(1)}, y^{(1)},\cdots y^{(n)})
= p(y^{(1)}, y^{(2)}, \cdots, y^{(n)}|\theta) = \prod_{i=1}^n p(y^{(i)}|\theta)

`}
  </Latex>
  <p>
    We can change <Latex>\theta</Latex> and observe how the likelihood changes. Lets
    assume we tossed the coin and got Heads, Heads, Tails, Heads and Tails.
  </p>
  <p class="blue">
    Parameter <Latex>\theta</Latex>: {theta}, Likelihood <Latex
      >{String.raw`\mathcal{L}`}</Latex
    >: {likelihood.toFixed(5)}
  </p>
  <Plot
    pointsData={[{ x: theta, y: likelihood }]}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: 0,
      maxX: 1,
      minY: 0.00001,
      maxY: 0.04,
      xLabel: "Theta",
      yLabel: "Likelihood",
      padding: { top: 20, right: 40, bottom: 40, left: 55 },
      xTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      yTicks: [0, 0.01, 0.02, 0.03, 0.04],
    }}
  />
  <Slider min={0} max={1} step={0.01} bind:value={theta} />
  <p>
    When you increase <Latex>\theta</Latex> by moving the slider to the right, you
    will notice that the likelihood will increase up to a point and start decreasing
    again. At roughly <Latex>\theta = 0.6</Latex> the likelihood will reach the maximum.
    This is called the <Highlight>maximum likelihood</Highlight> and the parameters
    that produce the maximum likelihood is exactly what we are looking for.
  </p>
  <p>
    In logistic regression we are looking for weights<Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and the bias <Latex>b</Latex> that maximize the likelihood, provided we are
    faced with the features matrix <Latex>{String.raw`\mathbf{X}`}</Latex> and the
    labels vector <Latex>{String.raw`\mathbf{y}`}</Latex>.
  </p>
  <Latex
    >{String.raw`\underset{\mathbf{w}, b}{\arg\max}  \mathcal{L(\mathbf{w}, b | \mathbf{X}, \mathbf{y})}`}</Latex
  >
  <p>
    Given a binary classification problem, we can use logistic regression and
    calculate the probability of a class <Latex>{String.raw`y^{(i)}`}</Latex> using
    the sigmoid function <Latex>\sigma</Latex>. Recognizing that the likelihood
    and the probability are the same, we can rewrite the likelihood in terms of
    probabilities.
  </p>
  <Latex
    >{String.raw`\mathcal{L(\mathbf{w}, b | \mathbf{X}, \mathbf{y})} = \prod_{i=1}^n \sigma^{y^{(i)}} (1 - \sigma)^{1 - y^{(i)}}`}</Latex
  >
  <p>
    In practice it is difficult to use gradient descent when we are dealing with
    a product. For once we would prefer to work with sums rather than products
    when taking derivatives. This makes the calculation easier, because the
    derivative of the sum is the sum of derivatives, which enables us to
    separately calculate the derivatives for individual samples. The derivative
    of a product on the other hand would involve the product rule, which would
    overcomplicate things. Additionally, when we calculate the product of small
    numbers, like probabilities, the number will get smaller and smaller and
    might get to a point where it would underflow. Instead in practice we
    maximize the natural log of the likelihood. This turns products into sums
    and exponents into products.
  </p>
  <Latex
    >{String.raw`\log \mathcal{L(\mathbf{w}, b | \mathbf{X}, \mathbf{y})} = \sum_{i=1}^n {y^{(i)}} \log(\sigma) + ({1 - y^{(i)}}) \log(1 - \sigma)`}</Latex
  >
  <p>
    A question that often occurs the first time you apply a logarithm to a
    function you want to optimize: "do the optimal values of <Latex
      >{String.raw`\mathbf{w}`}</Latex
    > and <Latex>b</Latex> change by applying the natural logarithm?" The answer
    is no. Below are depicted 2 functions, the original function <Latex
      >{String.raw`f(x) = x^2 + 0.5`}</Latex
    > and the transformed function <Latex
      >{String.raw`g(f(x)) = \log(f(x))`}</Latex
    >.
  </p>
  <Plot
    pathsData={transformData}
    config={{
      width: 500,
      height: 250,
      maxWidth: 1000,
      minX: -7,
      maxX: 7,
      minY: -1,
      maxY: 20,
      xLabel: "x",
      yLabel: "Output",
      padding: { top: 20, right: 40, bottom: 40, left: 60 },
      radius: 5,
      colors: [
        "var(--main-color-1)",
        "var(--main-color-2)",
        "var(--text-color)",
      ],
      xTicks: [],
      yTicks: [],
      numTicks: 5,
    }}
  />
  <p>
    The original function is the parabola at the top and the transformed
    function is the one at the bottom. As you can see the same x value of 0
    leads to the minimum function output. Therefore it does not matter that we
    optimize <Latex>{String.raw`\log(f(x))`}</Latex> instead of <Latex
      >f(x)</Latex
    >.
  </p>
  <p>
    Often we calculate the mean of the log likelihood. This is perfectly legal,
    because this procedure also does not change the parameters <Latex
      >{String.raw`\mathbf{w}`}</Latex
    >
    and <Latex>b</Latex> which maximize the log likelihood.
  </p>
  <Latex
    >{String.raw`\log \mathcal{L(\mathbf{w}, b | \mathbf{X}, \mathbf{y})} = \dfrac{1}{n} \sum_{i=1}^n {y^{(i)}} \log(\sigma) + ({1 - y^{(i)}}) \log(1 - \sigma)`}</Latex
  >
  <p>
    If we wanted to maximize the log likelihood function, we would need to apply
    gradient ascent. Most deep learning libraries only implement gradient
    descent though. Maximizing <Latex>f(x)</Latex> is the same as minimizing <Latex
      >-f(x)</Latex
    >. For that reason we change the sign of the log likelihood and minimize the
    negative log likelihood.
  </p>
  <Latex
    >{String.raw`- \log \mathcal{L(\mathbf{w}, b | \mathbf{X}, \mathbf{y})} = - \dfrac{1}{n} \sum_{i=1}^n {y^{(i)}} \log(\sigma) + ({1 - y^{(i)}}) \log(1 - \sigma)`}</Latex
  >
  <p>At this point you might have already made the following discovery.</p>
  <p class="info">
    Minimizing the cross-entropy equals maximizing the log likelihood.
  </p>
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
