<script>
  import Container from "$lib/Container.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import Circle from "$lib/plt/Circle.svelte";

  // table
  import Table from "$lib/base/table/Table.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  const valueHeader = ["Original Gradient", "Clipped Gradient"];
  const valueData = [
    [1, 1],
    [0.5, 0.5],
    [2, 1],
    [-3, -1],
  ];

  let valuePaths = [];
  function recalculateValue() {
    valuePaths = [];
    // original value
    let x = Math.random() * 6 - 3;
    let y = Math.random() * 6 - 3;

    valuePaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);

    //clip values
    if (x >= 1) {
      x = 1;
    } else if (x < -1) {
      x = -1;
    }
    if (y >= 1) {
      y = 1;
    } else if (y < -1) {
      y = -1;
    }

    valuePaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);
  }
  recalculateValue();

  let normPaths = [];
  function recalculateNorm() {
    normPaths = [];
    // original value
    let x = Math.random() * 6 - 3;
    let y = Math.random() * 6 - 3;

    normPaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);

    let norm = Math.sqrt(x ** 2 + y ** 2);
    if (norm > 1) {
      x = x / norm;
      y = y / norm;
    }

    normPaths.push([
      { x: 0, y: 0 },
      { x, y },
    ]);
  }
  recalculateNorm();
</script>

<svelte:head>
  <title>Gradient Clipping - World4AI</title>
  <meta
    name="description"
    content="Gradient clipping clips either individual gradient values or the norm of the gradient vector at a predetermined threshold, thereby reducing the likelihood of exploding gradients."
  />
</svelte:head>

<h1>Gradient Clipping</h1>
<div class="separator" />

<Container>
  <p>
    The exploding gradients problem arises when the gradients get larger and
    larger, until they get larger than the maximum permitted value for the
    tensor datatype.
  </p>
  <p>
    We could remedy the problem with a simple solution. We could determine the
    maximum allowed gradient and if the gradient value moves beyond that
    threshold we cut the gradient to the allowed value. The technique we just
    described is called <Highlight>gradient clipping</Highlight>, value clipping
    to be exact.
  </p>
  <p>
    The below table demonstrates how value clipping works in theory. We set the
    threshold value to 1 and if the absolute value of the gradient moves beyond
    the threshold, we clip it to the max value.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each valueHeader as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each valueData as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>
    Value clipping is problematic, because it basically changes the direction of
    gradient descent. Below is a simulation to demonstrate the problem. When you
    start the simulation, the gradient vector (dashed line) will start to move
    randomly in the 2d coordinate system. If one of of the two gradients is
    larger than one, we will clip that gradient to 1 and thus create a new
    gradient vector (red line). So if one gradient is 3 and the other is 1.5, we
    clip both to 1, thereby disregarding the relative magnitude of the vector
    components and changing the direction of the vector. This is not what we
    actually desire.
  </p>
  <ButtonContainer>
    <PlayButton f={recalculateValue} delta={800} />
  </ButtonContainer>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-3, 3]}
    range={[-3, 3]}
  >
    <Ticks
      xTicks={[-3, -2, -1, 0, 1, 2, 3]}
      yTicks={[-3, -2, -1, 0, 1, 2, 3]}
    />
    <Path
      data={[
        { x: 1, y: 1 },
        { x: -1, y: 1 },
        { x: -1, y: -1 },
        { x: 1, y: -1 },
        { x: 1, y: 1 },
      ]}
      color="var(--main-color-1)"
    />
    <Path data={valuePaths[1]} color="var(--main-color-1)" />
    <Path data={valuePaths[0]} strokeDashArray={[4, 4]} />
  </Plot>
  <p>
    A better solution is to use norm clipping. When we clip the norm of the
    gradient vector, we clip all the gradients proportionally, such that the
    direction remains the same.
  </p>
  <p>
    Below is a simulation of norm clipping. When the magnitude of the gradient
    vector is reduced to the threshold value, the direction remains unchanged.
  </p>
  <ButtonContainer>
    <PlayButton f={recalculateNorm} delta={800} />
  </ButtonContainer>
  <Plot
    width={500}
    height={500}
    maxWidth={500}
    domain={[-3, 3]}
    range={[-3, 3]}
  >
    <Ticks
      xTicks={[-3, -2, -1, 0, 1, 2, 3]}
      yTicks={[-3, -2, -1, 0, 1, 2, 3]}
    />
    <Circle data={[{ x: 0, y: 0 }]} color="none" radius="70" />
    <Path data={normPaths[0]} strokeDashArray={[4, 4]} />
    <Path data={normPaths[1]} color="var(--main-color-1)" />
  </Plot>
  <p>
    Norm clipping often feels like a hack, but it is actually quite practical.
    You might not be able to solve all your problems with gradient clipping, but
    it should be part of your toolbox.
  </p>
  <p>
    The implementation of gradient clipping is PyTorch is astonishingly simple.
    All we have to do is to add the following line of code after we call <code
      >loss.backward()</code
    >
    but before we call <code>optimizer.step()</code>.
  </p>
  <PythonCode
    code={`nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)`}
  />
  <p>
    The above line concatenates all parameter gradients into a single vector,
    calculates the norm for that vector and eventually clips the gradients
    in-place, if the norm is above 1.
  </p>
  <div class="separator" />
</Container>
