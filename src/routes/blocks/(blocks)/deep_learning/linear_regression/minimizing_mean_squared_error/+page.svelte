<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

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
  
  let var1 = new Value(500);
  var1._name = 'Branch 1';
  let var2 = new Value(196);
  var2._name = 'Branch 2';
  let sum = var1.add(var2);
  sum._name = 'Sum';
  let mse2 = sum.mul(0.5);
  mse2._name = 'MSE';
  mse2.backward();

  // code examples
  let code1 = `import torch
import sklearn.datasets as datasets`;
  let code2 = `X, y = datasets.make_regression(n_samples=100, n_features=2, n_informative=2, noise=0.01)`;
  let code3 = `X = torch.from_numpy(X).to(torch.float32);
y = torch.from_numpy(y).to(torch.float32).unsqueeze(1)`
  let code4 = `def init_weights():
    w = torch.randn(1, 2, requires_grad=True)
    b = torch.randn(1, 1, requires_grad=True)
    return w, b
w, b = init_weights()`;
  let code5 = `def print_all():
    print(f'Weight: {w.data}, Grad: {w.grad}')
    print(f'Bias: {b.data}, Grad: {b.grad}')
print_all()`;
  let code6 = `def forward(w, b):
    y_hat = X @ w.T + b
    return ((y - y_hat)**2).sum() / 100.0
mse = forward(w, b)`;
  let code7 = `print_all()`;
  let code8 = `mse.backward()
print_all()`;
  let code9 = `mse = forward(w, b)
mse.backward()
print_all()`;
  let code10 = `w.grad.zero_()
print(w.grad)`;
  let code11 = `lr = 0.1
w, b = init_weights()
for _ in range(10):
    # forward pass
    mse = forward(w, b)
    
    print(f'Mean squared error: {mse.data}')
    
    # backward pass
    mse.backward()
    
    # gradient descent
    with torch.inference_mode():
        w.data.sub_(w.grad * lr)
        b.data.sub_(b.grad * lr)
        w.grad.zero_()
        b.grad.zero_()`;
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
  <BackpropGraph graph={step3} maxWidth={350} width={450} height={910} />
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
  <BackpropGraph graph={step4} maxWidth={350} width={450} height={910} />
  <p>
    Once we have the gradients, we can apply the gradient descent algorithm. The
    below example iterates between constructing the graph, calculating the
    gradients and applying gradient descent, eventually leading to a mean
    squared error of 0.
  </p>
  <ButtonContainer>
    <PlayButton f={trainLoop} delta={200} />
  </ButtonContainer>
  <BackpropGraph graph={mse} maxWidth={350} width={450} height={910} />
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
    complicated. Let's for example assume that we have two samples. This creates two branches in our computational graph.
  </p>
  <BackpropGraph graph={mse2} maxWidth={350} height={500} width={450} />
  <p>The branch nr. 2 that amounts to 196.0 is essentially the same path that we calulated above. For the other branch we would have to do the same calculations that we did above, but using the features from the other sample. To finish the calculations of the mean squared error, we would have to sum up the two branches and calculate the average. When we initiate the backward pass, the gradients are propagated into the individual branches. You should remember the the same weights and the same bias exist in the different branches. That means that those parameters receive different gradient signals. In that case the gradients are accumulated through a sum. This is the same as saying: "the derivative of a sum is the sum of derivatives". You should also observe, that the mean squared error scales each of the gradient signals by the number of samples that we use for training. If we have two samples, each of the gradients is divided by 2 (or multiplied by 0.5).
  </p>
  <p>
    So the calculations remain very similar: construct the graph and distribute the
    gradients to the weight and bias using automatic differentiation. When we
    use the procedures described above it does not make a huge difference whether we use 1, 4 or 100 samples.
  </p>
  <p>
    To convince ourselves that automatic differentiation actually works, below we present
    the example from the last section, that we solve by implementing a custom
    autodiff package in JavaScript.
  </p>
  <ButtonContainer>
    <PlayButton f={train} delta={1} />
  </ButtonContainer>
  <Mse data={dataMse} w={w.data} b={b.data} />
  <div class="separator"></div>

  <h2>Autograd</h2>
  <p>As mentioned before, in modern deep learning we do not iterate over individual samples to construct a graph, but work with tensors to parallelize the computations. When we utilize any of the modern deep learning packages and use the provided tensor objects, we get parallelisation and automatic differentiation out of the box. We do not need to explicitly construct a graph and make sure that all nodes are connected. PyTorch for example has a built in automatic differentiation library, called <Highlight>autograd</Highlight>, so let's see how we can utilize the package. 
  </p>
  <PythonCode code={code1}></PythonCode>
  <p>
   We start by creating the features <Latex>{String.raw`\mathbf{X}`}</Latex> and the labels <Latex>{String.raw`\mathbf{y}`}</Latex>.</p>
  <PythonCode code={code2}></PythonCode>
  <p>We transorm the generated numpy arrays into PyTorch Tensors. For the labels Tensor we use the <code>unsqueeze(dim)</code> method. This adds an additional dimension, transforming the labels from a (100,) into a (100, 1) dimensional Tensor. This makes sure that the predictions that are generated in the the forward pass and the actual labels have identical dimensions.</p>
  <PythonCode code={code3}></PythonCode>
  <p>We generate the <code>init_weights()</code> function, which initializes the weights and the biases randomly using the standard normal distribution. This time around we set the <code>requires_grad</code> property to <code>True</code> in order to track the gradients. We didn't do that for the features and the label tensors, as those are fixed and should not be adjusted.</p>
  <PythonCode code={code4}></PythonCode>
  <p>For the sake of making our explanations easier let us introduce the <code>print_all()</code> function. This function makes use of the two imortant properties that each Tensor object posesses. The <code>data</code> and the <code>grad</code> property. Those properties are probaly self explanatory: data contains the actual values of a tensor, while grad contains the gradiens with respect to each value in the data list.</p>
  <PythonCode code={code5}></PythonCode>
  <pre class="text-sm">
    Weight: tensor([[-0.6779,  0.4228]]), Grad: None
    Bias: tensor([[0.2107]]), Grad: None
  </pre>
  <p>When we print the data and the grad right after initializing the tensors, the objects posess a randomized value, but gradients amount to <code>None</code>.</p>
  <PythonCode code={code6}></PythonCode>
  <p>Even when we calculate the mean squared error, by running through the forward pass, the gradients remain empty.</p>
  <PythonCode code={code7}></PythonCode>
  <pre class='text-sm'>
    Weight: tensor([[-0.6779,  0.4228]]), Grad: None
    Bias: tensor([[0.2107]]), Grad: None
  </pre>
  <p>To actually run the backward pass, we have to call the <code>backward()</code> method on the loss function. The gradients are always based on the tensor that initiated the backward pass. So if we run the backward pass on the mean squared error tensor, the gradients tell us how we should shift the weights and the bias to reduce the loss. This is exactly what we are looking for.</p>
  <PythonCode code={code8}></PythonCode>
  <pre class='text-sm'>
    Weight: tensor([[-0.6779,  0.4228]]), Grad: tensor([[-102.5140,  -98.1595]])
    Bias: tensor([[0.2107]]), Grad: tensor([[4.4512]])
  </pre>
  <p>If we run the forward and the backward passes again, you will notice, that the weights and the bias gradients are twice as large. Each time we calculate the gradients, the gradients are accumulated. The old gradient values are not erased, as one might assume.</p>
  <PythonCode code={code9}></PythonCode>
  <pre class='text-sm'>
    Weight: tensor([[-0.6779,  0.4228]]), Grad: tensor([[-205.0279, -196.3191]])
    Bias: tensor([[0.2107]]), Grad: tensor([[8.9024]])
  </pre>
  <p>Each time we are done with a gradient descent step, we should clear the gradients. We can do that by using the <code>zero_()</code> method, which zeroes out the gradients inplace.</p>
  <PythonCode code={code10}></PythonCode>
  <pre class='text-sm'>
    tensor([[0., 0.]])
  </pre>
  <p>Below we show the full implementation of gradiet descent. Most of the implementation was already discussed before, but the context manager <code>torch.inference_mode()</code> might be new to you. This part tells PyTorch to not include the following parts in the computational graph. The actual gradient descent step is not part of the forward pass and should therefore not be tracked.</p>
  <PythonCode code={code11}></PythonCode>
  <pre class='flex justify-center text-sm'>
Mean squared error: 6125.7783203125
Mean squared error: 4322.3662109375
Mean squared error: 3054.9150390625
Mean squared error: 2162.31787109375
Mean squared error: 1532.5537109375
Mean squared error: 1087.494873046875
Mean squared error: 772.5029907226562
Mean squared error: 549.2703857421875
Mean squared error: 390.8788757324219
Mean squared error: 278.37469482421875
  </pre>
  <p>We iterate over the forward pass, the backward pass and the gradient descent step for 10 iterations and the mean squared error decreases dramatically. This iteration process is called the <Highlight>training loop</Highlight> in deep learning lingo. We will encounter those loops over and over again.</p>
  <div class="separator" />
</Container>
