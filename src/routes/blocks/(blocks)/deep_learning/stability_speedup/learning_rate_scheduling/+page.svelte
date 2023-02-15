<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";
  import Title from "$lib/plt/Title.svelte";

  let data = [];
  for (let i = -8; i <= 8; i += 0.1) {
    data.push({ x: i, y: i ** 2 });
  }

  let xCoordinateFixedLR = 8;
  $: yCoordinateFixedLR = xCoordinateFixedLR ** 2;
  let alphaFixed = 0.1;
  let momentumFixed = 0;
  let prevYFixed = yCoordinateFixedLR;

  function gradientDescentStepFixed() {
    prevYFixed = yCoordinateFixedLR;
    let beta = 0.95;
    let grad = 2 * xCoordinateFixedLR;
    if (momentumFixed === 0) {
      momentumFixed = grad;
    }
    momentumFixed = momentumFixed * beta + grad * (1 - beta);
    xCoordinateFixedLR -= alphaFixed * momentumFixed;
  }

  let xCoordinateMovingLR = 8;
  $: yCoordinateMovingLR = xCoordinateMovingLR ** 2;
  let alphaMoving = 0.1;
  let momentumMoving = 0;
  let prevYMoving = yCoordinateMovingLR;

  function gradientDescentStepMoving() {
    if (xCoordinateMovingLR ** 2 > prevYMoving) {
      alphaMoving *= 0.88;
    }
    prevYMoving = yCoordinateMovingLR;
    let beta = 0.95;
    let grad = 2 * xCoordinateMovingLR;
    if (momentumMoving === 0) {
      momentumMoving = grad;
    }
    momentumMoving = momentumMoving * beta + grad * (1 - beta);
    xCoordinateMovingLR -= alphaMoving * momentumMoving;
  }
</script>

<svelte:head>
  <title>Learning Rate Scheduling - World4AI</title>
  <meta
    name="description"
    content="It is not always an easy task to finetune the learning rate. Learning rate schedulers are a common way to otpimize this hyperparameter, but changing the learning rate over time."
  />
</svelte:head>

<h1>Learning Rate Scheduling</h1>
<div class="separator" />

<Container>
  <p>
    There is probably no hyperparameter that is more important than the learning
    rate <Latex>\alpha</Latex>. If the learning rate is too high, we might
    overshood or oscilate. If the learning rate is too low, training might be
    too slow, or we might get stuck in some local minimum.
  </p>
  <p>
    In the example below for example we pick a learning rate that is relatively
    large. The gradient descent algorithm (with momentum) overshoots and keeps
    oscilating for a while, before settling on the minimum.
  </p>
  <ButtonContainer>
    <PlayButton f={gradientDescentStepFixed} delta={50} />
  </ButtonContainer>
  <Plot domain={[-8, 8]} range={[0, 60]}>
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 10, 20, 30, 40, 50, 60]}
    />
    <Path {data} />
    <Circle data={[{ x: xCoordinateFixedLR, y: yCoordinateFixedLR }]} />
    <Title text={`Constant Learning Rate ${alphaFixed.toFixed(2)}`} />
  </Plot>

  <p>
    It is possible, that a single constant rate is not the optimal solution.
    What if we start out with a relatively large learning rate to gain momentum
    at the beginning of trainig, but decrease the learning rate either over time
    or at specific events. In deep learning this is called <Highlight
      >learning rate decay</Highlight
    > or <Highlight>learning rate scheduling</Highlight>. There are dozens of
    different schedulers (see the
    <a
      href="https://pytorch.org/docs/stable/optim.html"
      target="_blank"
      rel="noreferrer">PyTorch documentation</a
    >
    for more info). You could for example decay the learing rate by subtracting a
    constant rate every <Latex>n</Latex> episodes. Or you could multiply the learning
    rate at the end of each epoch by a constant factor, for example
    <Latex>0.9</Latex>. Below we use a popular learning rate decay technique
    that is called <Highlight>reduce learning rate on plateau</Highlight>. Once
    a metric (like a loss) stops improving for certain amount of epochs we
    decrease the learning rate by a predetermined factor. Below we use this
    technique, which reduces the learning rate once the algorithm overshoots. It
    almost looks like the ball "glides" into the optimal value.
  </p>
  <ButtonContainer>
    <PlayButton f={gradientDescentStepMoving} delta={50} />
  </ButtonContainer>
  <Plot domain={[-8, 8]} range={[0, 60]}>
    <Ticks
      xTicks={[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]}
      yTicks={[0, 10, 20, 30, 40, 50, 60]}
    />
    <Path {data} />
    <Circle data={[{ x: xCoordinateMovingLR, y: yCoordinateMovingLR }]} />
    <Title text={`Variable Learning Rate ${alphaMoving.toFixed(3)}`} />
  </Plot>
  <p>
    Deep learning frameworks like PyTorch or Keras make it extremely easy to
    create learning rate schedulers. Usually it involves no more than 2-3 lines
    of code.
  </p>
  <p>
    Schedulers in PyTorch are located in <code>otpim.lr_scheduler</code>, in our
    example we pick <code>optim.lr_scheduler.ReduceLROnPlateau</code>. All
    schedulers take an optimizer as input. This is necessary because the
    learning rate is a part of an optimizer and the scheduler has to modify that
    paramter. The <code>patience</code> attribute is a ReduceLROnPlateau
    specific paramter that inidicates for how many epochs the performance metric
    (like cross-entropy) has to shrink in order to for the learning rate to be
    multiplied by the <code>factor</code> parameter of 0.1.
  </p>
  <PythonCode
    code={`model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1)`}
  />
  <p>
    We have to adjust the train function to introduce the scheduler logic. This
    function might for example look as below. Similar to the <code
      >optimizer.step()</code
    >
    method there is a
    <code>scheduler.step()</code> method. This function takes a performance measure
    lke the validation loss and adjusts the learning rate if necessary.
  </p>
  <PythonCode
    code={`def train(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler):
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (features, labels) in enumerate(train_dataloader):
            # switch to training mode
            model.train()
            # move features and labels to GPU
            features = features.view(-1, NUM_FEATURES).to(DEVICE)
            labels = labels.to(DEVICE)

            # ------ FORWARD PASS --------
            probs = model(features)

            # ------CALCULATE LOSS --------
            loss = criterion(probs, labels)

            # ------BACKPROPAGATION --------
            loss.backward()

            # ------GRADIENT DESCENT --------
            optimizer.step()

            # ------CLEAR GRADIENTS --------
            optimizer.zero_grad()

        # ------TRACK LOSS --------
        train_loss, train_acc = track_performance(train_dataloader, model, criterion)
        val_loss, val_acc = track_performance(val_dataloader, model, criterion)
        
        # ------ADJUST LEARNING RATE --------
        scheduler.step(val_loss)`}
  />
  <p>
    There are no hard rules what scheduler you need to use in what situation,
    but when you use PyTorch you need to always call <code
      >optimizer.step()</code
    >
    before you call <code>scheduler.step()</code>.
  </p>
  <div class="separator" />
</Container>
