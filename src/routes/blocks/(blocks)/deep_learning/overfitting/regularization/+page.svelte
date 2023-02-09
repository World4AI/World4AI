<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import Slider from "$lib/Slider.svelte";
  import L2Polynomial from "../_regularization/L2Polynomial.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Text from "$lib/plt/Text.svelte";

  import l1_overfitting from "./l1_overfitting.png";
  import l2_overfitting from "./l2_overfitting.png";

  // circles of different size
  function generateUnit(len) {
    let path = [];
    for (let i = 0; i <= Math.PI * 2; i += 0.01) {
      let x = Math.cos(i) * len;
      let y = Math.sin(i) * len;
      path.push({ x, y });
    }
    return path;
  }

  let unitPath = generateUnit(1);
  let vectorPath;
  let unitPoint;

  let circularAngle = 0.5;
  function moveCircular() {
    circularAngle += 0.1;
    let x = Math.cos(circularAngle);
    let y = Math.sin(circularAngle);
    vectorPath = [
      { x: 0, y: 0 },
      { x, y },
    ];
    unitPoint = [{ x, y }];
  }
  moveCircular();

  let circle1 = generateUnit(1);
  let circle2 = generateUnit(2);
  let circle3 = generateUnit(3);

  // draw equation of the form x_1 + x_2 = 3
  let infiniteSolutions = [];
  for (let i = -4; i <= 4; i++) {
    let x = i;
    let y = 3 - i;
    infiniteSolutions.push({ x, y });
  }

  let radius = 0.1;
  $: touchingCircle = generateUnit(radius);

  // diamonds of different size
  function generateDiamond(len = 1) {
    let path = [];
    for (let i = 0; i <= 1; i += 0.1) {
      let x = i * len;
      let y = len - x;
      path.push({ x, y });
    }
    for (let i = 1; i >= 0; i -= 0.1) {
      let x = i * len;
      let y = -(len - x);
      path.push({ x, y });
    }
    for (let i = 0; i <= 1; i += 0.1) {
      let x = -i * len;
      let y = -(len + x);
      path.push({ x, y });
    }
    for (let i = 1; i >= 0; i -= 0.1) {
      let x = -i * len;
      let y = len + x;
      path.push({ x, y });
    }
    return path;
  }

  // move point along the diamond
  let diamondPath = generateDiamond();
  let diamondVectorPath;
  let diamondPoint;
  let count = 0;
  function moveDiamond() {
    count = count % (diamondPath.length - 1);
    let x = diamondPath[count].x;
    let y = diamondPath[count].y;
    diamondPoint = [{ x, y }];
    diamondVectorPath = [{ x: 0, y: 0 }, ...diamondPoint];
    count += 1;
  }
  moveDiamond();
  let diamond1 = generateDiamond(1);
  let diamond2 = generateDiamond(2);
  let diamond3 = generateDiamond(3);

  // draw equation of the form 2x_1 + x_2 = 3
  let infiniteDiamondSolutions = [];

  for (let i = -4; i <= 4; i++) {
    let x = 2 * i;
    let y = 3 - i;
    infiniteDiamondSolutions.push({ x, y });
  }

  let size = 0.1;
  $: touchingDiamond = generateDiamond(size);

  const code1 = `LAMBDA = 0.01
def train_epoch(dataloader, model, criterion, optimizer):
    for batch_idx, (features, labels) in enumerate(train_dataloader):
        # move features and labels to GPU
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        # ------ FORWARD PASS --------
        output = model(features)

        # ------CALCULATE LOSS --------
        loss = criterion(output, labels)
        l2 = None
        for param in model.parameters():
            if l2 is None:
                l2 = param.pow(2).sum()
            else:
                l2 += param.pow(2).sum()
        
        loss += LAMBDA * l2

        # ------BACKPROPAGATION --------
        loss.backward()

        # ------GRADIENT DESCENT --------
        optimizer.step()

        # ------CLEAR GRADIENTS --------
        optimizer.zero_grad()`;

  const code2 = `model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=0.005)`;
  const code3 = `history = train(NUM_EPOCHS, train_dataloader, val_dataloader, model, criterion, optimizer)`;
  const output3 = `Epoch: 1/50|Train Loss: 0.4967 |Val Loss: 0.4781 |Train Acc: 0.8601 |Val Acc: 0.8630
Epoch: 10/50|Train Loss: 0.1641 |Val Loss: 0.1671 |Train Acc: 0.9574 |Val Acc: 0.9538
Epoch: 20/50|Train Loss: 0.1508 |Val Loss: 0.1548 |Train Acc: 0.9617 |Val Acc: 0.9595
Epoch: 30/50|Train Loss: 0.1363 |Val Loss: 0.1430 |Train Acc: 0.9660 |Val Acc: 0.9632
Epoch: 40/50|Train Loss: 0.1284 |Val Loss: 0.1369 |Train Acc: 0.9686 |Val Acc: 0.9655
Epoch: 50/50|Train Loss: 0.1300 |Val Loss: 0.1399 |Train Acc: 0.9681 |Val Acc: 0.9647
`;
  const code4 = `optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.001)`;
  const code5 = `LAMBDA = 0.01
def train_epoch(dataloader, model, criterion, optimizer):
    for batch_idx, (features, labels) in enumerate(train_dataloader):
        # move features and labels to GPU
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        # ------ FORWARD PASS --------
        output = model(features)

        # ------CALCULATE LOSS --------
        loss = criterion(output, labels)
        l1 = None
        for param in model.parameters():
            if l1 is None:
                l1 = param.abs().sum()
            else:
                l1 += param.abs().sum()
        
        loss += LAMBDA * l1

        # ------BACKPROPAGATION --------
        loss.backward()

        # ------GRADIENT DESCENT --------
        optimizer.step()

        # ------CLEAR GRADIENTS --------
        optimizer.zero_grad()`;
  const code6 = `model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=0.005)`;
  const code7 = `history = train(NUM_EPOCHS, train_dataloader, val_dataloader, model, criterion, optimizer)`;
  const output7 = `Epoch: 1/50|Train Loss: 1.1206 |Val Loss: 1.1060 |Train Acc: 0.6172 |Val Acc: 0.6270
Epoch: 10/50|Train Loss: 0.2200 |Val Loss: 0.2162 |Train Acc: 0.9383 |Val Acc: 0.9377
Epoch: 20/50|Train Loss: 0.2077 |Val Loss: 0.2087 |Train Acc: 0.9416 |Val Acc: 0.9405
Epoch: 30/50|Train Loss: 0.1744 |Val Loss: 0.1745 |Train Acc: 0.9507 |Val Acc: 0.9495
Epoch: 40/50|Train Loss: 0.1966 |Val Loss: 0.1966 |Train Acc: 0.9416 |Val Acc: 0.9417
Epoch: 50/50|Train Loss: 0.1604 |Val Loss: 0.1656 |Train Acc: 0.9541 |Val Acc: 0.9513
`;
  const code8 = `plot_history(history, 'l1_overfitting')`;
</script>

<svelte:head>
  <title>L1 and L2 Regularization - World4AI</title>
  <meta
    name="description"
    content="The L1 and L2 regularization techniques reduce overfitting by modifying the loss function. Both regularizers keep the size of the weights small, but the L1 loss also enforces sparsity."
  />
</svelte:head>

<h1>Regularization</h1>
<div class="separator" />
<Container>
  <p>
    The goal of regularization is to encourage simpler models. Simpler models
    can not fit the data exactly and are therefore less prone to overfitting.
    While there are several techniques to achieve that goal, in this section we
    focus on techniques that modify the loss function.
  </p>
  <p>
    We will start this section by reminding ourselves of a term usually learned
    in an introductory linear algebra course, the <Highlight>norm</Highlight>.
    The norm measures the distance of a vector
    <Latex>{String.raw`\mathbf{v}`}</Latex> from the origin. The p-norm (written
    <Latex>L_p</Latex>) for the vector <Latex>{String.raw`\mathbf{v}`}</Latex> is
    defined as <Latex
      >{String.raw`||\mathbf{v}||_p = \large\sqrt{\sum_{i=1}^n |v_i|^p}`}</Latex
    >. By changing the parameter <Latex>p</Latex> from 1 to infinity we get different
    types of norms. In machine learning and deep learning we are especially interested
    in the <Latex>{String.raw`L_1`}</Latex> and the<Latex
      >{String.raw`L_2`}</Latex
    > norm.
  </p>

  <div class="separator" />

  <h2>L2 Norm</h2>
  <p>
    The <Latex>L_2</Latex> norm also called Euclidean norm is the distance measure
    we are most familiar with. When we are given the vector <Latex
      >{String.raw`\begin{bmatrix}
        5 \\
        4
      \end{bmatrix}`}</Latex
    > for example we can regard the number 5 as the x coordinate and the numver 4
    as the y coordinate. As shown in the graph below this vector is essentially a
    hypotenuse of a right triangle, therefore we need to apply the Pythagorean theorem
    to calculate the length of the vector, <Latex
      >{String.raw`c^2 = \displaystyle\sqrt{a^2 + b^2}`}</Latex
    >.
  </p>

  <Plot width={500} height={500} maxWidth={500} domain={[0, 5]} range={[0, 5]}>
    <Path
      data={[
        { x: 0, y: 0 },
        { x: 5, y: 0 },
        { x: 5, y: 4 },
        { x: 0, y: 0 },
      ]}
    />
    <Circle data={[{ x: 5, y: 4 }]} />
    <Ticks xTicks={[0, 1, 2, 3, 4, 5]} yTicks={[0, 1, 2, 3, 4, 5]} />
    <Text text="a" x={2.5} y={0.2} fontSize={20} />
    <Text text="b" x={4.7} y={2} fontSize={20} />
    <Text text="c" x={2.5} y={2.5} fontSize={20} />
  </Plot>
  <p>
    While the the Pythagorean theorem is used to calculate the length of a
    vector on a 2 dimensional plane, the <Latex>L_2</Latex> norm generalizes the
    idea of length to
    <Latex>n</Latex> dimensions,
    <Latex
      >{String.raw`
||\mathbf{v}||_2 =\displaystyle \sqrt{\sum_{i=1}^n v_i^2}
      `}</Latex
    >.
  </p>
  <p />
  <p>
    Now let's assume we want to find all vectors on a two dimensional plane that
    have a specific <Latex>L_2</Latex> norm of size <Latex>l</Latex>. When we
    are given a specific vector length <Latex>l</Latex> such that <Latex
      >{String.raw`\displaystyle \sqrt{x_1^2 + x_2^2} = l`}</Latex
    >, we will find that there is whole set of vectors that satisfy that
    condition and that this set has a circular shape. In the interactive example
    below we assume that the norm is 1, <Latex
      >{String.raw`\displaystyle \sqrt{x_1^2 + x_2^2} = 1`}</Latex
    >. If we draw all the vectors with the norm of 1 we get a unit circle.
  </p>
  <ButtonContainer>
    <PlayButton f={moveCircular} delta={100} />
  </ButtonContainer>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-2, 2]}
    range={[-2, 2]}
  >
    <Path data={unitPath} />
    <Path data={vectorPath} />
    <Circle data={unitPoint} />
    <Ticks
      xTicks={[-2, -1, 0, 1, 2]}
      yTicks={[-2, -1, 0, 1, 2]}
      xOffset={-15}
      yOffset={20}
    />
    <XLabel text={"x_1"} type="latex" fontSize={15} />
    <YLabel text={"x_2"} type="latex" fontSize={15} x={0} />
  </Plot>
  <p>
    There is an infinite number of such circles, each corresponding to a
    different set of vectors with a particular <Latex>L_2</Latex> norm. The larger
    the radius, the larger the norm. Below we draw the circles that correspond to
    the <Latex>L_2</Latex> norm of 1, 2 and 3 respectively.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-4, 4]}
    range={[-4, 4]}
  >
    <Path data={circle1} />
    <Path data={circle2} />
    <Path data={circle3} />
    <Ticks
      xTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      yTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      xOffset={-15}
      yOffset={20}
    />
    <XLabel text={"x_1"} type="latex" fontSize={15} />
    <YLabel text={"x_2"} type="latex" fontSize={15} x={0} />
  </Plot>
  <p>
    Now let's try and anderstand how this visualization of a norm can be useful.
    Imagine we are given a single equation <Latex
      >{String.raw`x_1 + x_2 = 3`}</Latex
    >. This is an underdetermined system of equations, because we have just 1
    equation and 2 unknowns. That means that there is literally an infinite
    number of possible solutions. We can depict the whole set of solutions as a
    single line. Points on that line show combinations of <Latex>x_1</Latex> and
    <Latex>x_2</Latex> that sum up to 3.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-4, 4]}
    range={[-4, 4]}
  >
    <Path data={infiniteSolutions} />
    <Ticks
      xTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      yTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      xOffset={-15}
      yOffset={20}
    />
    <XLabel text={"x_1"} type="latex" fontSize={15} />
    <YLabel text={"x_2"} type="latex" fontSize={15} x={0} />
  </Plot>
  <p>
    Below we add the norm for the <Latex>x_1</Latex>, <Latex>x_2</Latex> vector.
    The slider controls the size of the <Latex>L_2</Latex> norm. When you keep increasing
    the norm you will realize that at the red point the circle is going to just barely
    touch the line. At this point we see the solution for <Latex
      >{String.raw`x_1 + x_2 = 3`}</Latex
    > that has the smallest <Latex>L2</Latex> norm out of all possible solutions.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-4, 4]}
    range={[-4, 4]}
  >
    <Path data={infiniteSolutions} />
    <Path data={touchingCircle} />
    <Circle data={[{ x: 1.5, y: 1.5 }]} />
    <Ticks
      xTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      yTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      xOffset={-15}
      yOffset={20}
    />
    <XLabel text={"x_1"} type="latex" fontSize={15} />
    <YLabel text={"x_2"} type="latex" fontSize={15} x={0} />
  </Plot>
  <Slider bind:value={radius} min="0.1" max="3" step="0.1" />
  <p>
    If we are able to find solutions that have a comparatively low <Latex
      >L_2</Latex
    > how does this apply to machine learning and why is this useful to avoid overfitting?
    We can add the <Highlight>squared</Highlight>
    <Latex>L_2</Latex> norm to the loss function as a regularizer. We do not use
    the <Latex>||L||_2</Latex> norm directly, but calculate the square of the norm,
    <Latex>||L||_2^2</Latex>, because the root makes the calculation of the
    derivative more complicarted than it needs to be.
  </p>
  <p>
    If we are dealing with the mean squared error for example, our new loss
    function looks as below.
  </p>
  <Latex
    >{String.raw`L=\dfrac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)} )^2 + \lambda \sum_j^m w_j^2`}</Latex
  >
  <p>
    The overall intention is to find the solution that reduces the mean squared
    error without creating large weights. When the size of one of the weights
    increses disproportionatly, the regularization term will increase and the
    loss function will rise sharply. In order to avoid a large loss, gradient
    descent will push the weights closer to 0. Therefore by using the
    regularization term we reduce the size of the weights and the overemphasis
    on any particular feature, thereby reducing the complexity of the model. The <Latex
      >\lambda</Latex
    > (lambda) is the hyperparameter that we can tune to determine how much emphasis
    we would like to put on the <Latex>L_2</Latex> norm. It is the lever that lets
    you control the size of the weights.
  </p>
  <p>
    Below we have the same model trained with and without the <Latex>L_2</Latex>
    regurlarization. You can move the slider to adjust the lambda. The higher the
    lambda, the simpler the model becomes and the more the curve looks like a straight
    line.
  </p>
  <L2Polynomial />
  <p>
    We can implement <Latex>L_2</Latex> regularization in PyTorch, by adding a couple
    more lines to our calculation of the loss function. Esentially we loop over all
    weights and biases, square those and calculate a sum. Autograd does the rest.
  </p>
  <PythonCode code={code1} />
  <PythonCode code={code2} />
  <p>Our regularization procedure does a fine job reducing overfitting.</p>
  <PythonCode code={code3} />
  <PythonCode code={output3} isOutput={true} />
  <img src={l2_overfitting} alt="Overfitting with L2 training" />
  <p>
    PyTorch actually provides a much easier way to implement <Latex>L_2</Latex> regularization.
    When you define your optimizer, you can pass the <code>weight_decay</code>
    parameter. This is essentially the <Latex>\lambda</Latex> from our equation above.
  </p>
  <PythonCode code={code4} />
  <div class="separator" />

  <h2>L1 Norm</h2>
  <p>
    The <Latex>L_1</Latex> norm, also called the Manhattan distance, simply adds
    absolute values of each element of the vector,
    <Latex
      >{String.raw`
      ||\mathbf{v}||_1 = \sum_{i=1}^n |v_i|
      `}</Latex
    >.
  </p>
  <p>
    This definition means essentially that when you want to move from the blue
    point to the red point, you do not take the direct route, but move along the
    axes.
  </p>
  <Plot width={500} height={500} maxWidth={450} domain={[0, 5]} range={[0, 5]}>
    <Path
      data={[
        { x: 0, y: 0 },
        { x: 5, y: 0 },
        { x: 5, y: 4 },
      ]}
    />
    <Circle data={[{ x: 0, y: 0 }]} color={"var(--main-color-2)"} />
    <Circle data={[{ x: 5, y: 4 }]} color={"var(--main-color-1)"} />
    <Ticks xTicks={[0, 1, 2, 3, 4, 5]} yTicks={[0, 1, 2, 3, 4, 5]} />
  </Plot>
  <p>
    We can make the same exercise we did with the <Latex>L_2</Latex> norm and imagine
    how the set of vectors looks like if we restrict the <Latex>L_1</Latex> norm
    to length <Latex>1</Latex>: <Latex>{String.raw`|x_1| + |x_2| = 1`}</Latex>.
    The result is a diamond shaped figure. All vectors on the ridge of the
    diamond have a <Latex>L_1</Latex> norm of exactly 1.
  </p>

  <ButtonContainer>
    <PlayButton f={moveDiamond} delta={100} />
  </ButtonContainer>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-2, 2]}
    range={[-2, 2]}
  >
    <Path data={diamondPath} />
    <Path data={diamondVectorPath} />
    <Circle data={diamondPoint} />
    <Ticks
      xTicks={[-2, -1, 0, 1, 2]}
      yTicks={[-2, -1, 0, 1, 2]}
      xOffset={-15}
      yOffset={20}
    />
    <XLabel text={"x_1"} type="latex" fontSize={15} />
    <YLabel text={"x_2"} type="latex" fontSize={15} x={0} />
  </Plot>
  <p>
    Different <Latex>L_1</Latex> norms in 2D produce diamonds of different sizes.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-4, 4]}
    range={[-4, 4]}
  >
    <Path data={diamond1} />
    <Path data={diamond2} />
    <Path data={diamond3} />
    <Ticks
      xTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      yTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      xOffset={-15}
      yOffset={20}
    />
    <XLabel text={"x_1"} type="latex" fontSize={15} />
    <YLabel text={"x_2"} type="latex" fontSize={15} x={0} />
  </Plot>
  <p>
    Below we are given an underdetermined system of equations <Latex
      >2x_1 + x_2 = 3</Latex
    > and we want to find a solution with the smallest <Latex>L_1</Latex> norm.
  </p>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-4, 4]}
    range={[-4, 4]}
  >
    <Path data={infiniteDiamondSolutions} />
    <Path data={touchingDiamond} />
    <Circle data={[{ x: 0, y: 3 }]} />
    <Ticks
      xTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      yTicks={[-4, -3, -2, -1, 0, 1, 2, 3, 4]}
      xOffset={-15}
      yOffset={20}
    />
    <XLabel text={"x_1"} type="latex" fontSize={15} />
    <YLabel text={"x_2"} type="latex" fontSize={15} x={0} />
  </Plot>
  <Slider bind:value={size} min="0.1" max="4" step="0.1" />
  <p>
    When you move the slider you will find the solution, where the diamond
    touches the line. This solution produces the vector with the lowest <Latex
      >L_1</Latex
    > norm. An important characteristic of the <Latex>L_1</Latex> norm is the so
    called <Highlight>sparse</Highlight> solution. The diamond has four sharp points.
    Each of those points corresponds to a vector where only one of the vector elements
    is not zero (this is also valid for more than two dimensions). That means that
    when the diomond touches the function, we are faced with a solution where the
    vector is mostly zero, a sparse vector.
  </p>
  <p>
    When we add the <Latex>L_1</Latex> regularization to the mean squared error,
    we are simultaneously reducing the mean squared error and reduce the <Latex
      >L_1</Latex
    > norm. Similar to <Latex>L_2</Latex>, the <Latex>L_1</Latex> regularization
    reduces overfitting by not letting the weights grow disproportionatly. Additionally
    the <Latex>L_1</Latex> norm tends to generate sparse weights. Most of the weights
    will correspond to 0.
  </p>
  <Latex
    >{String.raw`L=\frac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)} )^2 + \lambda \sum_j^m |w_j|`}</Latex
  >
  <p>
    We can implement <Latex>L_1</Latex> regularization, but adjusting our loss function
    slightly. The rest of the implementation is the same.
  </p>
  <PythonCode code={code5} />
  <PythonCode code={code6} />
  <PythonCode code={code7} />
  <PythonCode code={output7} isOutput={true} />
  <PythonCode code={code8} />
  <img src={l1_overfitting} alt="overfitting with l1 norm" />
  <div class="separator" />
</Container>
