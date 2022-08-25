<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";
  
  let references = [
    {
        author: "Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi",
        title: "You Only Look Once: Unified, Real-Time Object Detection",
        year: "2015",
    },
    {
        author: "Redmon, Joseph and Farhadi, Ali",
        title: "YOLO9000: Better, Faster, Stronger", year: "2016",
    },
    {
        author: "Redmon, Joseph and Farhadi, Ali",
        title: "YOLOv3: An Incremental Improvement",
        year: "2018",
    }
  ];

  // svg configuration
  const width = 500;
  const height = 500;
  
  let S = 7; // number of grids
  const lines = [];
  
  // vertical lines
  let wFraction = width / S;
  for (let i = 0; i<=S; i++) {
    let line = {x1: i*wFraction, y1: 0, x2: i*wFraction, y2: height}         
    lines.push(line);
  }
  // horizontal lines
  let hFraction = height / S;
  for (let i = 0; i<=S; i++) {
    let line = {x1: 0, y1: i * hFraction, x2: width, y2: i*hFraction }         
    lines.push(line);
  }

  // circle object
  let cx = 85;
  let cy = 110;

  // circle active cell
  $: circleActiveX = Math.floor(cx / wFraction) * wFraction;
  $: circleActiveY = Math.floor(cy / hFraction) * hFraction;

  // ellipse object
  let ecx = 350
  let ecy = 250

  // circle active cell
  let ellipseActiveX = Math.floor(ecx / wFraction) * wFraction;
  let ellipseActiveY = Math.floor(ecy / hFraction) * hFraction;

</script>

<svelte:head>
  <title>World4AI | Deep Learning | YOLO</title>
  <meta
    name="description"
    content="The YOLO algorithm (you only look once) divides each image into an Sxs grid and lets each of the grid cells decide if there is an object in that particular cell. The cnn network outputs a SxSx30 vector that contains information what of the cells contain objects, what classes these objects belong to and the coordinates of the bounding boxes." 
  />
</svelte:head>

<h1>YOLO</h1>
<div class="separator" />
<Container>
  <p>There are several different algorithms out there that try to deal with object detection. By far the most popular algorithm and the one we are going to focus on in this section belongs to the family of YOLO<InternalLink type="reference" id={1} /> (you only look once). The algirithm is very efficient, relatively easy to understand and can even be applied to videos in real time. We are going to primarily focus on the original implementation (also called YOLOv1), but be aware that over time many more improvements have been published: for example YOLOv2<InternalLink type="reference" id={2} /> and YOLOv3<InternalLink type="reference" id={3} />.</p>
  <p>YOLO divides each image into an SxS grid. In the original implementation the authors used a 7x7 grid, but depending on your hardware you might consider increasing or decreasing the number. Each cell can potentially produce a bounding box, but only those cells that contain a center of some object are responsible for producing a bounding box for that object.</p>
  <p>In the below image there are two objects (circle and ellipse) that the YOLO algorithm needs to classify and put a bounding box around. The red squares mark the cells that are responsible for outputting a bounding box. You can move the yellow circle with the sliders below and observe how the responsibility shifts from cell to cell once the center of the object leaves a cell.</p>
  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 {width} {height}">
      <!-- grid line -->
      {#each lines as line}
        <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} stroke="black" />
      {/each}

      <!-- active cell 1 -->
      <rect x={circleActiveX} y={circleActiveY} width={wFraction} height={hFraction} fill="var(--main-color-1)" stroke="black" />

      <!-- active cell 2 -->
      <rect x={ellipseActiveX} y={ellipseActiveY} width={wFraction} height={hFraction} fill="var(--main-color-1)" stroke="black" />

      <!-- object 1 -->
      <circle {cx} {cy} r={50} fill="var(--main-color-3)" stroke="black" opacity={0.5}/>
      <rect x={cx-50} y={cy-50} width={100} height={100} fill="none" stroke="black" />
      <circle {cx} {cy} r={3} fill="var(--main-color-1)" stroke="black"/>
      
      <!-- object 2 -->
      <ellipse cx={ecx} cy={ecy} rx="100" ry="50" fill="var(--main-color-2)" stroke="black" opacity={0.5} />
      <rect x={ecx-100} y={ecy-50} width={200} height={100} fill="none" stroke="black" />
      <circle cx={ecx} cy={ecy} r={3} fill="var(--main-color-1)" stroke="black"/>
    </svg>
  </SvgContainer>
  <div class="flex">
    <span>Circle X: </span><Slider bind:value={cx} min={0} max={width} />
  </div>
  <div class="flex">
    <span>Circle Y: </span><Slider bind:value={cy} min={0} max={height} />
  </div>
  <p>It is perfectly legal for the object (and the bounding box) to be outside of the cell that is drawing it, but only a single cell can be responsible for drawing a bounding box. Only the center of the object matters. Theoretically the algorithm could find up to 49 (7x7) objects in the image, but there is also a limitation to the YOLO algorithm. A cell can only draw a single bounding box. If it happens the the center point of two objects is located in the single cell, we are out of luck and will only be able to draw a single box. Practically this is hightly unlikely with a large enough grid.</p> <p>The YOLO algorithm is based on a convolutional neural network. We use 448x448 pixel images as input and apply 24 convolutional layers (mixed with max pooling) and 2 fully connected layers. The last fully connected layer produces 1470 outputs, which we can conveniently reshape into a 7x7x30 tensor. That shape implies that for each of the grid cells we produce a vector of length 30. The vector contains enough information to determine if there is an object in the cell, what type of the object it is and the coordinates of the bounding box.</p>

  <p>The first five elements in the vector contain the probability that there is an object in that cell <Latex>Pr</Latex>, the <Latex>x</Latex>, <Latex>y</Latex> coordinates of the bounding box and the width <Latex>w</Latex> and height <Latex>h</Latex> of the bounding box.</p>
  <div class="vector">
    <div class="cell">
      <Latex>Pr</Latex>      
    </div>
    <div class="cell">
      <Latex>x</Latex>      
    </div>
    <div class="cell">
      <Latex>y</Latex>      
    </div>
    <div class="cell">
      <Latex>w</Latex>      
    </div>
    <div class="cell">
      <Latex>h</Latex>      
    </div>
    <div class="cell">
      <Latex>...</Latex>      
    </div>
  </div>
  <p>All five elements are scaled between 0 and 1 by applying a sigmoid activation function. We do that in order to be able to deal with images of varying dimensions. The <Latex>x</Latex> and <Latex>y</Latex> coordinates represent the relative offset of the center of an object from the upper left corner of the cell. For example if an object is exaclty in the middle of the cell, the coordinates would be 0.5 and 0.5 respectively. The width and the height on the other hand represent dimensions relative to the whole image. For example if the image is 500px by 500px and the bounding box is 100px by 50px, we would face a width of 0.2 and a height of 0.1.</p>
  
  <div class="vector">
    <div class="cell">
      <Latex>...</Latex>      
    </div>
    <div class="cell">
      <Latex>Pr_2</Latex>      
    </div>
    <div class="cell">
      <Latex>x_2</Latex>      
    </div>
    <div class="cell">
      <Latex>y_2</Latex>      
    </div>
    <div class="cell">
      <Latex>w_2</Latex>      
    </div>
    <div class="cell">
      <Latex>h_2</Latex>      
    </div>
    <div class="cell">
      <Latex>...</Latex>      
    </div>
  </div>

  <p>The next five elements contain similar values for a second bounding box.  In the example from before the circle needed a bounding box that was symmetrical, while the ellipse required an elongated box. By training a model with two boxes the YOLO algorithm encourages the specialization of each bounding box. In each of the cells only one of the bounding boxes can be drawn at the same time. When we train the model for example, we only calculate the loss for the bounding boss with the highest IOU and disregard the other.</p>
  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 {width} {height}">
      <!-- grid line -->
      {#each lines as line}
        <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} stroke="black" />
      {/each}

      <!-- object 1 -->
      <circle {cx} {cy} r={50} fill="var(--main-color-3)" stroke="black"/>
      <rect x={cx-50} y={cy-50} width={100} height={100} fill="none" stroke="black" />
      <circle {cx} {cy} r={3} fill="var(--main-color-1)" stroke="black"/>
      
      <!-- object 2 -->
      <ellipse cx={ecx} cy={ecy} rx="100" ry="50" fill="var(--main-color-2)" stroke="black" />
      <rect x={ecx-100} y={ecy-50} width={200} height={100} fill="none" stroke="black" />
      <circle cx={ecx} cy={ecy} r={3} fill="var(--main-color-1)" stroke="black"/>
    </svg>
  </SvgContainer>

  <p>The last 20 elements contain probabilities for the box to contain any of the 20 possible classes. The original YOLO algorithm was trained on the <a href="http://host.robots.ox.ac.uk/pascal/VOC/" target="_blank">PASCAL Visual Object Classes (VOC)</a> dataset that contained 20 classes like person, bird, aeroplan and bottle. This part of the vector is required for the object classification task.</p>
  <div class="vector">
    <div class="cell">
      <Latex>...</Latex>      
    </div>
    <div class="cell">
      <Latex>P_(c_1)</Latex>      
    </div>
    <div class="cell">
      <Latex>P_(c_2)</Latex>      
    </div>
    <div class="cell">
      <Latex>...</Latex>      
    </div>
    <div class="cell">
      <Latex>{String.raw`P(c_{20})`}</Latex> 
    </div>
  </div>
  <p>The last remaining question is "how do we train this model?". In essence the authors used the mean squared error for all 7x7 cells and 30 vector elements, but they added a couple of adjustments to make the training more stable.</p>
  <Latex>
    {String.raw`
      \lambda_{coord}\sum^{S^2}_{i=0}\sum^{B}_{j=0}{1}^{obj}_{i j} \Big[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \Big] \\ 
      + \\
      \lambda_{coord}\sum^{S^2}_{i=0}\sum^{B}_{j=0}{1}^{obj}_{i j} \Big[(\sqrt{w_i} - \sqrt{\hat{w_i}})^2 + (\sqrt{h_i} - \sqrt{\hat{h_i}})^2 \Big] \\     
      + \\
     \sum^{S^2}_{i=0}\sum^{B}_{j=0} {1}^{obj}_{i j} \Big( C_i + \hat{C}_i \Big)^2 \\     
      + \\
     \lambda_{noobj} \sum^{S^2}_{i=0}\sum^{B}_{j=0} {1}^{noobj}_{i j} \Big( C_i + \hat{C}_i \Big)^2 \\     
      + \\
     \sum^{S^2}_{i=0} {1}^{noobj}_{i} \sum_{c \in classes} \Big( p_i(c) + \hat{p}_i(c) \Big)^2 \\     
    `}
  </Latex>
  <p>The loss function might look complicated, but remember, that we are still dealing with mean squared error. <Latex>S^2</Latex> is the number of grid cells and <Latex>B</Latex> is the number of bounding boxes. <Latex>x_i</Latex> and <Latex>y_i</Latex> are just coordinates of the center point for the cell <Latex>i</Latex>. The variables <Latex>w_i</Latex> and <Latex>h_i</Latex> are the width and height of the <Latex>i</Latex>'th cell respectively. We take the square root of the width and the height to emphasize that small errors in large boxes matter less than small errors in small boxes. The expression <Latex>{String.raw`1^{obs}_{i j}`}</Latex> is 1 when there is an object in cell <Latex>i</Latex> for bounding box <Latex>j</Latex> and 0 otherwise. That means that we only calculate the loss for the coordinates of the bounding box for those cells that actually contain an object. We decide which of the two boxes is the box with an object by comparing the IOUs of the two and taking the one with the highest value. We scale the loss of the bounding box by <Latex>{String.raw`\lambda_{coord}`}</Latex> in order to put a higher emphasise on the bounding box. The value is equal to 5 in the YOLO paper. <Latex>C_i</Latex> is the confidence that determines if there is actually an object in the cell. In our case this is just a value between 0 and 1 depicting the probability. If there is no object in the cell, we scale the conficence loss by <Latex>{String.raw`\lambda_{noobj}`}</Latex>, which corresponds to 0.5. This is done in order to avoid overemphasis on empty cells. Finally the loss for the class probability values <Latex>{String.raw`p_i(c)`}</Latex> are calculated only for the cells with actual objects.</p>
  <p>Many of the concepts that we have covered so far might seem foregn to you rigth now. If you want to comprehend the algorithm in more detail, you will need to work through the practical section.</p>
</Container>
<Footer {references} />


<style>
  .flex {
    display: flex;
    align-items: center;
  }

  .flex span {
    width: 100px;
  }

  .vector {
    display: flex;
    gap: 10px;
    justify-content: center;
    align-items: center;
  }
  .cell {
    width: 50px;
    height: 50px;
    border: 1px solid black;
    display: flex;
    justify-content: center;
    align-items: center;
  }

</style>
