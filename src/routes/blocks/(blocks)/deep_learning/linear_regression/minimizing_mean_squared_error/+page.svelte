<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import Alert from "$lib/Alert.svelte";

  import Mse from "../_loss/Mse.svelte";
  import BackpropGraph from "$lib/backprop/BackpropGraph.svelte";
  import { Value } from "$lib/Network.js";

  // gradient descent mse
  const dataMse = [
    { x: 5, y: 20 },
    { x: 10, y: 40 },
    { x: 35, y: 15 },
    { x: 45, y: 59 },
  ];

  let w = new Value(1);
  let b = new Value(1);
  let mseAlpha = 0.001;

  function train() {
    let mse = new Value(0);
    dataMse.forEach((point) => {
      let pred = w.mul(point.x).add(b);
      mse = mse.add(pred.sub(new Value(point.y)).pow(2));
    });
    mse = mse.div(4);
    mse.backward();
    w.data -= mseAlpha * w.grad;
    b.data -= mseAlpha * b.grad;
    w.grad = 0;
    b.grad = 0;
  }

  let mse;
  function linearRegression() {
    let w = new Value(1);
    w._name = "Weight: w";
    let b = new Value(1);
    b._name = "Bias b";
    let mse;
    function train() {
      w.grad = 0;
      b.grad = 0;
      mse = null;
      let x = new Value(dataMse[0].x);
      x._name = "Feature: x";
      let scaled = w.mul(x);
      scaled._name = "w * x";
      let pred = scaled.add(b);
      pred._name = "Prediction";
      let target = new Value(dataMse[0].y);
      target._name = "Target";
      let error = target.sub(pred);
      error._name = "Error";
      mse = error.pow(2);
      mse._name = "MSE";
      mse.backward();
      return { mse, w, b };
    }
    return train;
  }

  let trainFn = linearRegression();
  function trainLoop() {
    let w;
    let b;
    let obj = trainFn();
    ({ mse, w, b } = obj);
    w.data -= 0.001 * w.grad;
    b.data -= 0.001 * b.grad;
  }

  let rest;
  ({ mse, ...rest } = trainFn());

  // explain autodiff step by step
  let step1;
  let step2;
  let step3;
  let step4;
  function steps() {
    let w = new Value(1);
    w._name = "Weight: w";
    let b = new Value(1);
    b._name = "Bias b";
    let mse;
    w.grad = 0;
    b.grad = 0;
    let x = new Value(dataMse[0].x);
    x._name = "Feature: x";
    let scaled = w.mul(x);
    scaled._name = "w * x";
    step1 = JSON.parse(JSON.stringify(scaled));

    let pred = scaled.add(b);
    pred._name = "Prediction";
    step2 = JSON.parse(JSON.stringify(pred));

    let target = new Value(dataMse[0].y);
    target._name = "Target";
    let error = target.sub(pred);
    error._name = "Error";
    mse = error.pow(2);
    mse._name = "MSE";
    step3 = JSON.parse(JSON.stringify(mse));
    mse.backward();
    step4 = JSON.parse(JSON.stringify(mse));
  }

  steps();

  // several samples
  let mse2;
  function linearRegression2() {
    let w = new Value(1);
    let b = new Value(1);
    let mse;
    w.grad = 0;
    b.grad = 0;
    mse = null;
    dataMse.forEach((point) => {
      let pred = w.mul(point.x).add(b);

      if (!mse) {
        mse = pred.sub(new Value(point.y)).pow(2);
      } else {
        mse = mse.add(pred.sub(new Value(point.y)).pow(2));
      }
    });
    mse = mse.div(dataMse.length);
    mse.backward();
    mse2 = mse;
  }
  linearRegression2();
</script>

<svelte:head>
  <title>Minimizing Mean Squared Error- World4AI</title>
  <meta
    name="description"
    content="In linear regression we can calculate the gradients of the weights and bias by construction a computational graph and applying automatic differentiation. Those gradients can be used in the gradient descent procedure to find optimal weights and biases."
  />
</svelte:head>

<h1>Minimizing MSE</h1>
<div class="separator" />

<Container>
  <h2>Single Training Sample</h2>
  <p>
    Let us remind ourselves that our goal is to minimize the mean squared error,
    by tweaking the weight vector <Latex>{String.raw`\mathbf{w}`}</Latex> and the
    bias scalar <Latex>b</Latex> using gradient descent.
    <Alert type="info">
      <Latex
        >{String.raw`
        MSE=\dfrac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)})^2 \\

        `}</Latex
      >
      <div class="my-2" />
      <Latex
        >{String.raw`
             \hat{y}^{(i)} = \mathbf{x}^{(i)} \mathbf{w}^T + b 
     `}</Latex
      >
    </Alert>
    To make our journey easier, let us for now assume that we have one single feature
    <Latex>x</Latex> and one single training sample. That reduces the mean squared
    error to a much simpler form:
    <Latex>{String.raw` MSE=(y - [xw + b])^2`}</Latex>. We will extend our
    calculations once we have covered the basics.
  </p>
  <p>
    The computation of gradients in the deep learning world relies heavily on
    the <Highlight>chain rule</Highlight>.
  </p>
  <Alert type="info">
    If we have a composite function <Latex>z(y(x))</Latex>, we can calculate the
    derivative of <Latex>z</Latex> with respect of <Latex>x</Latex> by applying the
    chain rule.
    <div class="mb-2" />
    <Latex
      >{String.raw`
        \dfrac{dz}{dx} = \dfrac{dz}{dy} \dfrac{dy}{dx} = \dfrac{dz}{\xcancel{dy}} \dfrac{\xcancel{dy}}{dx} = \dfrac{dz}{dx} 
    `}</Latex
    >
  </Alert>
  <p>
    The mean squared error <Latex>{String.raw`(y - [xw + b])^2`}</Latex> is a great
    example of a composite function. We start by calculating the scaled feature
    <Latex>s</Latex>, where <Latex>s = w\times x</Latex>. We use s as an input
    and calculate the linear regression prediction, <Latex
      >{String.raw`\hat{y} = s + b`}</Latex
    >. Using <Latex>{String.raw`\hat{y}`}</Latex> as input, we can calculate the
    error <Latex>e</Latex> as the difference between the prediction and the true
    target <Latex>y</Latex>, <Latex>{String.raw`e = y - \hat{y}`}</Latex>.
    Finally we calculate the mean squared error for one single training example, <Latex
      >mse = e^2</Latex
    >.
  </p>
  <p>
    If we utilize the chain rule, we can calculate the derivative of the mean
    squared error with respect to the bias <Latex>b</Latex> and the weight <Latex
      >w</Latex
    > by multiplying the intermediary derivatives.
  </p>
  <div class="flex justify-center items-center flex-col">
    <Latex
      >{String.raw`\dfrac{\partial mse}{\partial b} = \dfrac{\partial mse}{\partial e} \times \dfrac{\partial e}{\partial \hat{y}} \times \dfrac{\partial \hat{y}}{\partial b}`}</Latex
    >
    <div />
    <Latex
      >{String.raw`\dfrac{\partial mse}{dw} = \dfrac{\partial mse}{\partial e} \times \dfrac{\partial e}{\partial \hat{y}} \times \dfrac{\partial \hat{y}}{\partial s} \times \dfrac{\partial s}{\partial w}`}</Latex
    >
  </div>
  <Alert type="info">
    For didactic reasons we are focusing on linear regression with a single
    feature, but for the most part the calcultion with more than one feature
    would be the same.
    <div class="mb-2" />
    <Latex
      >{String.raw`\dfrac{\partial mse}{\partial w_j} = \dfrac{\partial mse}{\partial e} \times \dfrac{\partial e}{\partial \hat{y}} \times \dfrac{\partial \hat{y}}{\partial s} \times \dfrac{\partial s}{\partial w_j}`}</Latex
    >
    <div class="mb-2" />
    In that case we have to calculate as many partial derivatives, as there are features.
  </Alert>
  <p>
    Calculating those intermediary derivatives is relatively straightforward,
    using basic rules of calculus.
  </p>
  <div class="flex justify-center items-center flex-col">
    <Latex>{String.raw`\dfrac{\partial mse}{\partial e} = 2e`}</Latex>
    <div class="mb-1" />
    <Latex>{String.raw`\dfrac{\partial e}{\partial \hat{y}} = -1`}</Latex>
    <div class="mb-1" />
    <Latex>{String.raw`\dfrac{\partial \hat{y}}{\partial \hat{b}} = 1`}</Latex>
    <div class="mb-1" />
    <Latex>{String.raw`\dfrac{\partial \hat{y}}{\partial \hat{s}} = 1`}</Latex>
    <div class="mb-1" />
    <Latex>{String.raw`\dfrac{\partial s}{\partial w} = x`}</Latex>
  </div>
  <p>
    Using the chain rule we can easily calculate the derivatives with respect to
    the weight and bias.
  </p>
  <div class="flex justify-center items-center flex-col">
    <Latex
      >{String.raw`
  \begin{aligned} 
  \dfrac{\partial mse}{\partial b} &= 2e * (-1) * 1 \\
  &= -2(y - \hat{y})  \\
  &= -2(y - (wx + b)) 
  \end{aligned}
  `}</Latex
    >
    <div />
    <Latex
      >{String.raw`
  \begin{aligned} 
  \dfrac{\partial mse}{\partial b} &= 2e * (-1) * 1 * x \\
  &= -2x(y - \hat{y})  \\
  &= -2x(y - (wx + b)) 
  \end{aligned}
  `}</Latex
    >
  </div>
  <p>
    Once we have the gradients, the gradient descent algorithm works as
    expected.
  </p>
  <Alert type="info">
    <Latex
      >{String.raw`\mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w `}</Latex
    >
    <div class="mb-2" />
    <Latex
      >{String.raw`b_{t+1} \coloneqq b_t - \alpha \dfrac{\partial}{\partial b} `}</Latex
    >
  </Alert>
  <div class="separator" />

  <h2>Computationial Graph</h2>
  <p>
    We have mentioned before, that the chain rule plays an integral role in deep
    learning, but what we have covered so far has probably not made it clear why
    it is so essential.
  </p>
  <Alert type="info">
    We can construct a so called <Highlight>computational graph</Highlight> and use
    the chain rule for <Highlight>automatic differentiation</Highlight>.
  </Alert>
  <p>
    So let's construct the computational graph for the mean squared error step by
    step and see how automatic differentiation looks like.
  </p>
  <p>
    A computational graph basically tracks all atomic calculatios and their
    results in a tree like structure. We start out the calculation of MSE by
    creating the weight node <Latex>w</Latex> and the feature node <Latex
      >x</Latex
    >, but at that point those two variables are not connected to each other
    yet. Once we calculate <Latex>w*x</Latex>, we connect the two and the result
    is a new node, that lies on a higher level.
  </p>
  <BackpropGraph graph={step1} maxWidth={300} height={300} width={280} />
  <p>
    The yellow box represents the operations (like additions and
    multiplications), blue boxes contains the values of the node, while the red
    boxes contain the derivatives. We assume a feature of 5 and the weight of 1
    at the beginning, so the resulting node has the value of 5. At this point in
    time we have not calculated any derivatives yet, so all derivatives will
    amount to 0 for now.
  </p>
  <p>
    Next we create the bias <Latex>b</Latex> node with the initial value of 1 and
    add the node to the previously calculated node. Once again those nodes are reflected
    in the computational graph.
  </p>
  <BackpropGraph graph={step2} />
  <p>
    We keep creating new nodes, and connect them to previous generated nodes and
    the graph keeps growing. We do that until we reach the final node, in our
    case the mean squared error.
  </p>
  <BackpropGraph graph={step3} maxWidth={250} width={300} height={910} />
  <p>
    Once we have created our comutational graph, we can start calculating the
    gradients and applying the chain rule. This time around we start with the
    last node and go all the way to the nodes we would like to adjust: the
    weigth and the bias.
  </p>
  <p>
    Remember that we want to calculate the derivatives of MSE with respect to
    the weight and the bias. To achieve that we need to calculate the
    intermediary derivatives and apply the chain rule along the way. First we
    calculate the derivative of MSE with respect to itself. Basically we are
    asking ourselves, how much does the MSE change when we increase the MSE by
    1. As expected the result is just 1.
  </p>
  <BackpropGraph graph={step4} maxWidth={150} width={150} height={100} />
  <p>
    This might seem to be an unnecessary step, but the chain rule in the
    computational graph relies on multiplying intermediary derivatives by the
    derivative of the above node and if we left the value of 0, the algorithm
    would not have worked.
  </p>
  <p>
    Next we go a level below and calculate the local derivative of the error
    with respect to MSE. The derivative of <Latex>e^2</Latex> is just <Latex
      >2 \times error</Latex
    >, wich is 28. We apply the chain rule and multiply 28 by the derivative
    value of the above node and end up with 28.00.
  </p>
  <BackpropGraph graph={step4} maxWidth={150} width={150} height={300} />
  <p>
    We keep going down and calculate and calculate the next intermediary
    derivatives. This time around we face two nodes. The target node and the
    prediction node, where <Latex>Error = Target - Prediction</Latex>. The
    intermediary derivative of the error node with respect to the target is just
    1. The intermediary derivative of the error with respect to the prediction
    node is -1. Multiplying the intermediary derivatives with the derivative
    from the above node yields 28 and -28 respectively. These are the
    derivatives of the mean squared error with respect to the target and the
    prediction.
  </p>
  <BackpropGraph graph={step4} maxWidth={300} width={300} height={500} />
  <p>
    If we continue doing these calculations we will eventually end up with the
    below graph.
  </p>
  <BackpropGraph graph={step4} maxWidth={250} width={300} height={910} />
  <p>
    Once we have the gradients, we can apply the gradient descent algorithm. The
    below example iterates between constructing the graph, calculating the
    gradients and applying gradient descent, eventually leading to a mean
    squared error of 0.
  </p>
  <ButtonContainer>
    <PlayButton f={trainLoop} delta={200} />
  </ButtonContainer>
  <BackpropGraph graph={mse} maxWidth={250} width={300} height={910} />
  <p>
    The best part about the calculations above is that we do not have to track
    the nodes manually or to calculate the gradients ourselves. The connections
    between the nodes happen automatically. When we do any operations on these
    node objects, the computational graph is adjusted automatically to reflect
    the operations. The same goes for the calculation of the gradients, thus the
    name autodiff. Because the local nodes operations are usually very common
    operations like additions, multiplications and other common operations that
    are known to the deep learning community, we know how to calculate the
    intermediary derivatives. This behaviour is implemented in all deep learning
    packages and we do need to think about these things explicitly.
  </p>
  <p>
    There are several more advantages to utilizing a computation graph. For
    once, this allows us to easily calculate the gradients of arbitrary
    expressions, even if the expressions are extremely complex. As you can
    imagine, neural networks are such expressions, but if you only think in
    terms of local node gradients and the gradients that come from the node
    above, the calculations are rather simple to think about. Secondly, often we
    can reuse many gradient calculations, as these are used as inputs in
    multiple locations. For example the prediction node derivative is moved down
    to the bias and the scaled value and we only need to calculate the above
    derivative once. This is especially important for neural networks, as this
    algorithm allows us to efficiently distribute gradients, thus saving a lot
    of computational time.
  </p>
  <p>At this point we should mention, that in deep learning the graph construction phase and the gradient calculation phase have distinct names, that you will hear over and over again. The so called <Highlight>forward pass</Highlight> is essentially the graph constructon phase. In the <Highlight>backward pass</Highlight> we start to propagate the gradients from the the mean squared error to the weights and bias using automatic differentiation. This algorithm of finding the gradients of individual nodes, by utilizing the chain rule and a computational graph is also called <Highlight>backpropagation</Highlight>. Backpropagation is the bread and butter of modern deep learning.
  </p>

  <div class="separator" />

  <h2>Multiple Training Samples</h2>
  <p>
    As it turns out making a jump from one to several samples is not that
    complicated. You should remember from calculus that the derivative of a sum
    is the sum of derivatives. In other words in order to calculate the gradient
    of the mean squared error, we need to calculate the individual gradients for
    each sample and calculate the mean.
  </p>
  <Latex
    >{String.raw`
    \begin{aligned}
    & \mathbf{\nabla}_{w} = \dfrac{1}{n} \sum_i^n\mathbf{\nabla}^{(i)}_w \\
    & \dfrac{\partial}{\partial b} = \dfrac{1}{n}\sum^n_i\dfrac{\partial}{\partial b}^{(i)}
    \end{aligned}
    `}</Latex
  >
  <p>
    The computation graph below deals with just four samples, but it already
    becomes hard to visualize the procedure.
  </p>
  <BackpropGraph graph={mse2} maxWidth={500} width={1300} height={1800} />
  <p>
    Yet calculations remain very similar: construct the graph and distribute the
    gradients to the weight and bias using automatic differentiation. When we
    use the procedures described above it does not make a huge difference
    whether we use 1, 4 or 100 samples.
  </p>
  <p>
    To convince ourselves that the procedure actually works, below we present
    the example from the last section, that we solve by implementing a custom
    autodiff package in JavaScript.
  </p>
  <ButtonContainer>
    <PlayButton f={train} delta={1} />
  </ButtonContainer>
  <Mse data={dataMse} w={w.data} b={b.data} />
  <p>As mentioned before, in modern deep learning we do not iterate over individual samples to construct a graph, but work with tensors to parallelize the computations. When we utilize any of the modern deep learning packages and use the provided tensor objects, we get parallelisation and automatic differentiation out of the box. We do not need to explicitly construct a graph and make sure that all nodes are connected. We will see shortly how this can be accomplished with PyTorch.</p>
  <div class="separator" />

  <h2>Batch Gradient Descent</h2>
  <p>
    The approach of using the whole dataset to calculate the gradient is called <Highlight
      >batch</Highlight
    > gradient descent. Using the whole dataset has the advantage that we get a good
    estimation for the gradients, yet in many cases batch gradient descent is not
    used in practice. We often have to deal with datasets consisting of thousands
    of features and millions of samples. It is not possible to load all that data
    on the GPU's. Even if it was possible, it would take a lot of time to calculate
    the gradients for all the samples in order to take just a single training step.
    The alternatives described below are more practical and usually converge a lot
    faster.
  </p>
  <div class="separator" />
  <h2>Stochastic Gradient Descent</h2>
  <p>
    In stochastic gradient descent we introduce some stochasticity by shuffling
    the dataset randomly and using one sample at a time to calculate the
    gradient and to take a gradient descent step until we have used all samples
    in the dataset. This period of time, in which we exhaust all samples in the
    training dataset is called an <Highlight>epoch</Highlight>. After each epoch
    we reshuffle the data and start over. The advantage of stochastic gradient
    descent is that we do not have to wait for the calculation of gradients for
    all samples, but in the process we lose the advantages of parallelization
    that we get with batch gradient descent.
  </p>
  <p>
    When we calculate the gradient based on one sample the calculation is going
    to be off. By iterating over the whole dataset the sum of the directions is
    going to move the weights and biases towards the optimum. In fact this
    behaviour is often seen as advantageous, because theoretically the imprecise
    gradient could potentially push a variable from a local minimum.
  </p>
  <div class="separator" />

  <h2>Mini-Batch Gradient Descent</h2>
  <p>
    Mini-batch gradient descent combines the advantages of the stochastic and
    batch gradient descent. At the start of each epoch the dataset is shuffled
    randomly, but insdead of using one sample at a time in mini-batch gradient
    descent several samples are taken. Similar to the learning rate the size of
    the mini-batch is a hyperparameter and needs to be determined by the
    developer. Usually the size is calculated as a power of 2, for example 32,
    64, 128 and so on. You just need to remember that the batch needs to fit
    into the memory of your graphics card.
  </p>
  <p>
    Mini-batch gradient descent can be parallelized, because we use several
    samples at a time. Additionally it has the advantage that theoretically our
    training dataset can be as large as we want it to be.
  </p>
  <div class="separator" />

  <h2>Explicit Solution for Linear Regression</h2>
  <p>
    If you have taken a statistics course, you might remember, that there is an
    explicit solution to the linear regression problem, which does not involve
    gradient descent.
  </p>
  <Latex
    >{String.raw`\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}`}</Latex
  >
  <p>
    While it is true that we could use that equation to find the exact weights
    and bias that minimize the mean squared error, this approach does not scale
    well. If you look at the equation, you will notice that the calculation of
    the inverse of a matrix is required, which would slow down the calculation
    significantly as the number of training samples grows. In deep learning,
    where millions and millions of samples are required, this is not a feasible
    solution.
  </p>
  <p>
    Even if computation was not a major bottleneck, neural networks do not
    provide an explicit solution, therefore we are dependent on gradient
    descent.
  </p>
  <div class="separator" />
</Container>
