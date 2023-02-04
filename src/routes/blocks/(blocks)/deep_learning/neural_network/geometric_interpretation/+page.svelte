<script>
  import Container from "$lib/Container.svelte";
  import Highlight from '$lib/Highlight.svelte';
  import Transformation from "./_geometric/Transformation.svelte";
  import Latex from "$lib/Latex.svelte";
  import Alert from "$lib/Alert.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";

  import PlayButton from "$lib/button/PlayButton.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte"; 
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import XLabel from "$lib/plt/XLabel.svelte"; 
  import YLabel from "$lib/plt/YLabel.svelte"; 
  import Circle from "$lib/plt/Circle.svelte"; 
  import Rectangle from "$lib/plt/Rectangle.svelte";

  import { Value, Layer } from "$lib/Network.js";

  export let inputData = [
    { x: -1, y: 1 },
    { x: -1, y: -1 },
    { x: 1, y: -1 },
    { x: 1, y: 1 },
  ];
  export let inputData2 = [
    { x: -1, y: 1 },
    { x: -2, y: 0 },
    { x: -1, y: -1 },
    { x: 1, y: -1 },
    { x: 2, y: 0 },
    { x: 1, y: 1 },
  ];


  const layers = [
    {
      title: "Features",
      nodes: [
        { value: "x_1", class: "fill-gray-200" },
        { value: "x_2", class: "fill-gray-200" },
      ],
    },
    {
      title: "Hidden Layer 1",
      nodes: [
        { value: "a_{11}", class: "fill-blue-400" },
        { value: "a_{12}", class: "fill-blue-400" },
        { value: "a_{13}", class: "fill-blue-400" },
        { value: "a_{14}", class: "fill-blue-400" },
      ],
    },
    {
      title: "Hidden Layer 2",
      nodes: [
        { value: "a_{21}", class: "fill-yellow-400" },
        { value: "a_{22}", class: "fill-yellow-400" },
      ],
    },
    {
      title: "Output",
      nodes: [
        { value: "o", class: "fill-blue-400" },
      ],
    },
  ];
// create the data to draw the svg
  let pointsData = [[], []];
  let radius = [0.45, 0.25];
  let centerX = 0.5;
  let centerY = 0.5;
  let numPerCategory = 200;

  let Xs = [];
  let ys = [];
  for (let i = 0; i < radius.length; i++) {
    for (let point = 0; point < numPerCategory; point++) {
      //data for drawing
      let angle = 2 * Math.PI * Math.random();
      let r = radius[i];
      let x = r * Math.cos(angle) + centerX;
      let y = r * Math.sin(angle) + centerY;
      pointsData[i].push({ x, y });

      //data for training
      Xs.push([x, y]);
      ys.push(i);
    }
  }

  // determine the x and y coordinates that are going to be used for heatmap
  let numbers = 50;
  let heatmapCoordinates = [];
  for (let i = 0; i < numbers; i++) {
    for (let j = 0; j < numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let coordinate = [];
      coordinate.push(x);
      coordinate.push(y);
      heatmapCoordinates.push(coordinate);
    }
  }

  const alpha = 1;
  let heatmapData = [[],[]];
  let hiddenPointsData = [[], []];
  let lossOutput = 0;

  function train() {
    // use layers instead of mlp to access the hidden layers
    let layer1 = new Layer(2, 4);
    let layer2 = new Layer(4, 2);
    let layer3 = new Layer(2, 1)
    let loss = new Value(0);

    function step() { 
      //reset hidden coordinates
      hiddenPointsData = [[], []];
      //clear gradients
      layer1.zeroGrad();
      layer2.zeroGrad();
      layer3.zeroGrad();
      for (let i = 0; i < Xs.length; i++) {

        let out1 = layer1.forward(Xs[i]);
        let out2 = layer2.forward(out1);
        let out = layer3.forward(out2);
        //cross-entropy
        if (ys[i] === 0) {
          let one = new Value(1);
          loss = loss.add(one.sub(out).log());
        } else if (ys[i] === 1) {
          loss = loss.add(out.log());
        }

        //fill the hidden coordinates
        let x = out2[0].data;
        let y = out2[1].data;
        hiddenPointsData[ys[i]].push({x, y});
      }

      //calculate cross entropy
      loss = loss.neg().div(Xs.length);
      lossOutput = loss.data; 

      //backprop
      loss.backward();
      //gradient descent
      layer1.parameters().forEach((param) => {
        param.data -= alpha * param.grad;
      });
      layer2.parameters().forEach((param) => {
        param.data -= alpha * param.grad;
      });
      layer3.parameters().forEach((param) => {
        param.data -= alpha * param.grad;
      });
      //reset loss and accuracy
      loss = new Value(0);

      // create heatmap
      let class0 = [];
      let class1 = [];
      heatmapCoordinates.forEach((coordinates) => {
        let out1 = layer1.forward(coordinates);
        let out2 = layer2.forward(out1);
        let pred = layer3.forward(out2);
        if (pred.data < 0.5) { 
          class0.push({x : coordinates[0], y: coordinates[1]});
        } else {
          class1.push({x : coordinates[0], y: coordinates[1]});
        }
      })
      heatmapData = [];
      heatmapData.push(class0);
      heatmapData.push(class1);
    }
    return step;
  }
  let takeStep = train();

</script>

<svelte:head>
  <title>Deep Learning Geometric Interpretation - World4AI</title>
  <meta
    name="description"
    content="Neural networks transform the inputs by scaling, translating and rotating the data with matrix multiplication. Additionally matrix multiplications can move the hidden features between different dimensions to better solve the task. Activation functions squish the data to provide solutions for non linear problesm. The last layer is linearly separable."
  />
</svelte:head>

<h1>Geometric Interpretation</h1>
<div class="separator" />

<Container>
  <p>
    So far we have discussed how a computational graph works and how we can use autodiff packages to solve nonlinear problems. Yet we are still missing a crucial component that will allow us to understand the workings of neural networks. We need to answer the following question. <Highlight>How does a solution to a nonlinear problem looks geometrically?</Highlight>
  </p>
  <p>
    To get to that understanding, first we have to understand, that a neural network is a series of transormations. Multiplying the features with a matrix is a linear transformation, adding a bias is a translation and applying an activation function squishes the data. Each of these transformations accomplishes a different task and can be interpreted visually. Additionally we can stack several layers in a neural network, thus creating a transformation composition.
  </p>
  <Alert type='info'>
    A neural network is a composition of transformations.
  </Alert>
  <p>To demonstrate the visual interpretation of a transformation we will utilize the following matrix of features with 2 features and 4 samples.</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
        \mathbf{X} =
        \begin{bmatrix}
      -1 & 1 \\ 
      -1 & -1 \\
      1 &-1 \\
      1 & 1 \\
        \end{bmatrix}
          `}</Latex
    >
  </div>
  <p>The four samples build a square in a 2d coordinate system.</p>

  <Transformation animated={false} {inputData}/>
  <p>
    We will start with matrix multiplications.
  </p>
  <Alert type='info'>
    A matrix multiplication is a linear transformation. 
  </Alert>
  <p>While there is a formal definition of linear transformations, we could use a somewhat loose definition that you can use as a mental model. In a linear transformation parallel lines remain parallel and the origin does not move. So the four lines of the square above will remain parallel lines after the linear transformation.</p>
  <p>
    We can using linear transformations by multiplying the features matrix <Latex
      >{String.raw`\mathbf{X}`}</Latex
    > by the weight matrix 
    <Latex>{String.raw`\mathbf{W}^T`}</Latex>, our transformation matrix.
    Depending on the contents of <Latex>{String.raw`\mathbf{W}^T`}</Latex> different
    types of transformations are produced. The weight matrix is going to be a 2x2 matrix for now. That way we are using 2 features per sample as input and generate 2 transformed features per sample as output.
  </p>
   <p>The identity matrix is the easiest matrix to understand.</p>
  <Latex>
    {String.raw`
    \mathbf{W}^T =
    \begin{bmatrix}
    1 & 0 \\
    0 & 1 \\
    \end{bmatrix}`}
  </Latex>
  <p>Applying this transformation keeps the original matrix.</p>
  <p>
    If we change the values of the identity matrix slightly, we scale the original square. The matrix 
    <Latex
      >{String.raw`
  \mathbf{W}^T =
  \begin{bmatrix}
  2 & 0 \\
  0 & 1 \\
  \end{bmatrix}
  `}</Latex
    > for example scales the input square in the x direction by a factor of 2.
  </p>
  <Transformation
    {inputData}
    matrix={[
      [2, 0],
      [0, 1],
    ]}
  />
  <p>The matrix 
  <Latex>{String.raw`
  \mathbf{W}^T =
  \begin{bmatrix}
  1 & 0 \\
  0 & 0.5 \\
  \end{bmatrix}
  `}</Latex> on the other hand scales the matrix in the y direction by a factor of 0.5.
  </p>
  <Transformation
    {inputData}
    matrix={[
      [1, 0],
      [0, 0.5],
    ]}
  />
  <p>So far we have used only one diagonal of the matrix to scale the square. The other diagonal can be used for the so called sheer operation. When we use the below matrix
  <Latex>{String.raw`
  \mathbf{W}^T =
  \begin{bmatrix}
  1 & 0 \\
  1 & 1 \\
  \end{bmatrix}
  `}</Latex> for example, the top and the bottom lines are moved right and left respectively.</p>
  <Transformation
    {inputData}
    matrix={[
      [1, 0],
      [1, 1],
    ]}
  />
  <p>The matrix 
  <Latex>{String.raw`
  \mathbf{W}^T =
  \begin{bmatrix}
  1 & 1 \\
  0 & 1 \\
  \end{bmatrix}
  `}</Latex> on the other hand move the left and the right lines to the bottom or top respectively.</p>
  <Transformation
    {inputData}
    matrix={[
      [1, 1],
      [0, 1],
    ]}
  />
  <p>
    We can combine scaling and sheering to achieve interesting transformations. The matrix
    <Latex>
      {String.raw`
      \mathbf{W}^T =
      \begin{bmatrix}
      \cos(1) & -\sin(1) \\
      \sin(1) & \cos(1) \\
      \end{bmatrix}
      `}
    </Latex> for example rotates the data.
  </p>
  <Transformation
    {inputData}
    matrix={[
      [Math.cos(1), -Math.sin(1)],
      [Math.sin(1), Math.cos(1)],
    ]}
  />
  <p>Next let's look at the visual interpretation of the bias.</p>
  <Alert type='info'>
    Bias addition is a translation. 
  </Alert>
  <p>A bias allows us to translate the data. That means that each point is equally moved. The vector
    <Latex>
      {String.raw`
      \mathbf{b} =
      \begin{bmatrix}
      1  \\
      0 \\
      \end{bmatrix}
      `}
    </Latex> would move all points in the x direction by 1.
  </p>
  <Transformation
    {inputData}
    vector={[1, 0]}
  />
  <p>The vector
    <Latex>
      {String.raw`
      \mathbf{b} =
      \begin{bmatrix}
      0  \\
      1 \\
      \end{bmatrix}
      `}
    </Latex> on the other hand, moves all points by 1 in the y direction.
  </p>
  <Transformation
    {inputData}
    vector={[0, 1]}
  />
  <p>A translation is not a linear transformation. If we apply a linear transformation, that induces rotation, the zero point remains intact after the transformation.</p>
  <Transformation
    inputData={[{x:0, y:0}, {x:1, y:1}, {x:-1, y:1}]}
    matrix={[
      [Math.cos(1), -Math.sin(1)],
      [Math.sin(1), Math.cos(1)],
    ]}
  />
  <p>A translation on the other hand moves the origin.</p>
  <Transformation
    inputData={[{x:0, y:0}, {x:1, y:1}, {x:-1, y:1}]}
    vector={[1, 1]}
  />
  <p>A neural network combines a liner transformation with a translation. In linear algebra this transformation combination is called <Highlight>affine transformation</Highlight>.</p>
  <p>Let's finally move to activation functions.</p>
  <Alert type='info'>
    A nonlinear activation function squishes the data.
  </Alert>
  <p>
    We can imagine the nonlinear transformations as some sort of "squishing", where the activation function limits the data to a certain range. The sigmoid that we have utilized so far
    pushes the vectors into a 1 by 1 box.
  </p>
  <Transformation
    inputData={inputData2}
    activation="sigmoid"
    showText={false}
  />
  <p>The ReLU activation function is even wilder. The function turns negative numbers into zeros and leaves positive numbers untouched. With a ReLU parallel lines do not necessarily stay parallel.</p>
  <Transformation
    inputData={inputData2}
    activation="relu"
    showText={false}
  />
  <p>There is a final remark we would like to make in regards with linear transformations. So far we have used a 2x2 weight matrix for our linear transformations. We made this in order to keep the number of dimensions constant. We took in two features and produced two neurons. That way we could visualize the results in a 2d plane. If on the other hand we used a 2x3 matrix, the transformation would have pushed the features into 3d space. In deep learning we change the amount of dimensions all the time by changing the number of neurons from layer to layer. Sometimes the network can find a better solution in a different dimension.</p>
  <p>
    So what exactly does a neural network try to achieve throught those
    transformations? We are going to use a slightly different architecture to solve our circular data problem. The architecture below was not picked randomly, but to show some magic that is hidden under the hood of a neural network.
  </p>
  <NeuralNetwork {layers} height={140} padding={{right: 10, left: 0}} />
  <p>
    Let us remember that logistic regression is able to deal with classification
    problems, but only if the data is linearly separable. The last layer of the
    above neural network looks exacly like logistic regression with two input features. That must mean, that the neural network is somehow able to extract features through linear and nonlinear transformations, that are linearly separable.
  </p>
  <p>
    The example below shows how the neural network learns those transformations.
    On one side you can see the original inputs with the learned decision
    boundary, on the other side are the two extracted features that are used as
    inputs into the output layer. When the neural network has learned to
    separate the two circles, that means that the two features from the last
    hidden layer are linearly separable. Start the example and observe the
    learning process. At the beginning the hidden features are clustered
    together, but after a while you will notice that you could separate the
    different colored circles by a single line.
  </p>
  <ButtonContainer>
    <PlayButton f={takeStep} delta={0}/>
  </ButtonContainer>
  <span>Cross-Entropy: {lossOutput.toFixed(6)}<span>
  <div class="flex flex-col md:flex-row">
     <Plot
       width={500}
       height={500}
       maxWidth={600}
       domain={[0, 1]}
       range={[0, 1]}
     >
       <Ticks
         xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
         yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
         xOffset={-15}
         yOffset={15}
       />
       <Rectangle data={heatmapData[0]} size={9} color="var(--main-color-3)" />
       <Rectangle data={heatmapData[1]} size={9} color="var(--main-color-4)" />
       <Circle data={pointsData[0]} />
       <Circle data={pointsData[1]} color="var(--main-color-2)" />
       <XLabel text="Feature 1" fontSize={15} />
       <YLabel text="Feature 2" fontSize={15} />
     </Plot>
     <Plot
       width={500}
       height={500}
       maxWidth={600}
       domain={[0, 1]}
       range={[0, 1]}
     >
       <Ticks
         xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
         yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
         xOffset={-15}
         yOffset={15}
       />
       <Circle data={hiddenPointsData[0]} />
       <Circle data={hiddenPointsData[1]} color="var(--main-color-2)" />
       <XLabel text="Hidden Feature 1" fontSize={15} />
       <YLabel text="Hidden Feature 2" fontSize={15} />
     </Plot>
  </div>
  <p>
    It is not always clear how the neural network does those transformations,
    but we could use the example above to get some intuition for the process. If
    you look at the original circular data again you might notice something
    peculiar. Imagine the data is actually located in 3d space and you are
    looking at the data from above. Now imagine that the blue and the red dots
    are located on different heights (z-axis). Wouldn't that mean that you could
    construct a 2d plane in 3d space to linearly separate the data? Yes it
    would. The first hidden layer of our neural network transforms the 2d data
    into 4d data. Afterwards we move the processed features back into 2d space.
  </p>
  <p>
    Modern neural networks have hundreds or thousands of dimensions and hidden
    layers and we can not visualize the hidden features to get a better feel for
    what the neural network does. But generally speaking we can state the
    folllowing.
  </p>
  <Alert type="info">
    Affine transformations move, scale, rotate the data and move it between
    different dimensions. Activation functions squish or restraint the data to
    deal with nonlinearity. The last layers contain the hidden features, that
    can be linearly separated to solve a particular problem.
  </Alert>

  <p>
    Try to keep this intuition in mind while you move forward with your studies.
    It is easy to forget.
  </p>
  <div class="separator" />
</Container>

