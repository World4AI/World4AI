<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Object Detection</title>
  <meta
    name="description"
    content="Unlike image classification tasks, object detection has to classify and localize several objects at the same time within a single image. The variable number of objects makes the task a lot more difficult, but several families of algorithms like YOLO, have benn established to effectively deal with those problems, even in real time." 
  />
</svelte:head>

<h1>Object Detection</h1>
<div class="separator" />

<Container>
  <p>In the previous section of this chapter we have studied object classification. While image classification is an importnant task, computer vision is much more. Think about what tasks you can perform by looking at objects in your visual field. Not only can you tell what types of objects are located in the image, but you have no trouble identifying the exact location of the objects. This task is especially important, because one of the goals of artificial intelligence is to construct agents that can interact with our physical world. For that purpose they will need to learn to locate objects in order to avoid them or to interract with them. A similar case can be made for autonomous driving. A self-driving car needs to locate other vehicles, street signs, pedestirans and so on. Computer vision relies heavily on object detection.</p>
  <p>Below we can observe two computer vision tasks applied to the same image. In the first image we apply image classification techniques and classify the object as the triangle. In the second task additionally to classifying the image, we draw a bounding box around our object. This task is called <Highlight>object localization</Highlight>.</p>

  <div class="flex">
    <SvgContainer maxWidth="500px"> 
      <svg viewBox="0 0 500 200">
        <polygon points="250,50 300,150 200,150" fill="var(--main-color-3)" stroke="black" stroke-width="2" />
        <text x=10 y=20>Class: Triangle</text>
        <text x=10 y=42>Probability: 98%</text>
      </svg>
    </SvgContainer>
    
    <SvgContainer maxWidth="500px"> 
      <svg viewBox="0 0 500 200">
        <polygon points="250,50 300,150 200,150" fill="var(--main-color-3)" stroke="black" stroke-width="2" />
        <text x=10 y=20>Class: Triangle</text>
        <text x=10 y=42>Probability: 98%</text>
        <rect x=195 y=45 width=110 height=110 stroke-width=2 fill="none" stroke="var(--main-color-1)" />
      </svg>
    </SvgContainer>
  </div>
  <p>In order to localize the object, our model has to return some additional information. The bounding box needs two coordinates to be drawn: the (x, y) cooridnate of the upper left corner and the (x, y) coordinate of the bottom right corner.</p>
  <SvgContainer maxWidth="500px"> 
     <svg viewBox="0 0 500 200">
       <polygon points="250,50 300,150 200,150" fill="var(--main-color-3)" stroke="black" stroke-width="2" />
       <rect x=195 y=45 width=110 height=110 stroke-width=2 fill="none" stroke="var(--main-color-1)" />
       <circle cx=195 cy=45 r=4/>
       <circle cx=305 cy=155 r=4/>
     </svg>
  </SvgContainer>
  <p>Compared to object localization, <Highlight>object detection </Highlight> is a much harder task. We have to classify several objects at the same time, while localizing each object individually. And the number of objects in an image is variable and unknown during inference.</p>
  <SvgContainer maxWidth="500px"> 
    <svg viewBox="0 0 500 200">
      <polygon points="70,70 20,150 120,150" fill="var(--main-color-4)" stroke="black" stroke-width="1" />
      <text x=10 y=20>Class: Triangle</text>
      <text x=10 y=42>Probability: 98%</text>
      <rect x=15 y=65 width=110 height=90 stroke-width=1 fill="none" stroke="var(--main-color-1)" />
      <text x=300 y=70>Class: Circle</text>
      <text x=300 y=90>Probability: 92%</text>
      <circle cx=400 cy=150 r=30 stroke="black" fill="var(--main-color-3)" />
      <rect x=365 y=115 width=70 height=70 stroke-width=1 fill="none" stroke="var(--main-color-1)" />
    </svg>
  </SvgContainer>
  <div class="separator" />
</Container>

<style>
  svg {
    border: 1px solid black;
  }

  text {
    font-size: 20px;
  }

  .flex {
    display: flex;
    gap: 10px;
  }

  @media(max-width: 786px) {
    .flex {
      flex-direction: column;
    }
  }
</style>
