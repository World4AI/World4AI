<script>
  import Container from "$lib/Container.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Plus from "$lib/diagram/Plus.svelte";
  import Circle from "$lib/diagram/Circle.svelte";

  // button
  import PlayButton from "$lib/button/PlayButton.svelte";

  // coordinates of the circles in the fc diagram
  let fcCircle1x = 10;
  let fcCircle2x = 10;
  let fcCircle1y = 50;
  let fcCircle2y = 150;

  function simulateFc() {
    fcCircle1x+= 1;      
    fcCircle2x+= 1;      
    if (fcCircle1x >= 130 && fcCircle1y < 100.5){
      fcCircle1y+=0.45;
      fcCircle2y-=0.45;
    }

    if (fcCircle1x > 225){
      fcCircle1y = 100;
      fcCircle2y = 100;
    }

    if (fcCircle1x >=350) {
      fcCircle1x = 10;
      fcCircle2x = 10;
      fcCircle1y = 50;
      fcCircle2y = 150;
    }
  }

  // coordinates of the circles in the rnn diagram
  let rnnCircle1x = 10;
  let rnnCircle2x = 10;
  let rnnCircle1y = 50;
  let rnnCircle2y = 150;
  let loops = 3;
  let loop = 0;

  let turnAround = false;
  function simulateRnn() {

    // move forward
    if (!turnAround) {
      // move forward from start to declining arrow
      rnnCircle1x+= 1;      
      rnnCircle2x+= 1;      
  
      //  when you reach a declining arrow, start moving in the y direction as well
      if (rnnCircle1x >= 130 && rnnCircle1y < 100.5){
        rnnCircle1y+=0.45;
        rnnCircle2y-=0.45;
      }
      
      // join the two circles
      if (rnnCircle1x > 225){
        rnnCircle1y = 100;
        rnnCircle2y = 100;
      }

      if (rnnCircle1x >= 250 && loop < loops) {
        turnAround = true; 
      }
      // finished
      if (rnnCircle1x >=350) {
        rnnCircle1x = 10;
        rnnCircle2x = 10;
        rnnCircle1y = 50;
        rnnCircle2y = 150;
        loop = 0;
    }
    } 
    // loop around
    else {
      // move below  
      if (rnnCircle1y < 190 && rnnCircle1x >= 250) {
        rnnCircle1y+= 1;      
        rnnCircle2y+= 1;      
      } else if (rnnCircle1y >= 190 && rnnCircle1x > 10) {
        // move left
        rnnCircle1x-= 1;      
        rnnCircle2x-= 1;      
      } else if (rnnCircle1x <= 10 && rnnCircle1y > 150) {
        // move top
        rnnCircle1y-= 1;      
        rnnCircle2y-= 1;      
      } else {
        turnAround = false;
        rnnCircle1x = 10;
        rnnCircle2x = 10;
        rnnCircle1y = 50;
        rnnCircle2y = 150;
        loop+=1;
      }
    }
  }
</script>

<h1>The Fundamentals of Recurrent Neural Networks</h1>
<div class="separator" />

<Container>
  <!-- fc diagram -->
  <PlayButton f={simulateFc} delta={10}/>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 400 200">
      <!-- connections -->
      <Arrow strokeWidth=2 data={[{x: 5, y:50}, {x:65, y: 50}]} />
      <Arrow strokeWidth=2 data={[{x: 5, y:150}, {x:65, y: 150}]} />
      <Arrow strokeWidth=2 data={[{x: 130, y:50}, {x:215, y: 90}]} />
      <Arrow strokeWidth=2 data={[{x: 130, y:150}, {x:215, y: 110}]} />
      <Arrow strokeWidth=2 data={[{x: 280, y:100}, {x:380, y: 100}]} />

      <!-- neurons -->
      <Block x=100 y=50 width=50 height=50 color="var(--main-color-4)" />
      <Block x=100 y=150 width=50 height=50 color="var(--main-color-4)" />
      <Block x=250 y=100 width=50 height=50 color="var(--main-color-3)" />

      <!-- moving data -->
      <Circle x={fcCircle1x} y={fcCircle1y} r=5 color="var(--main-color-1)" />
      <Circle x={fcCircle2x} y={fcCircle2y} r=5 color="var(--main-color-1)" />
    </svg>
  </SvgContainer>

  <!-- rnn diagram -->
  <PlayButton f={simulateRnn} delta={10}/>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 400 250">
      {#each Array(4) as _, idx}
        <Block x={50 + idx*40} y=30 width=20 height=20 text={idx+1} fontSize={15} color={loop === idx ? "var(--main-color-1)" : "var(--main-color-4)"} />
      {/each}
      <g transform="translate(0 50)">
        <!-- connections -->
        <Arrow strokeWidth=2 data={[{x: 5, y:50}, {x:65, y: 50}]} />
        <Arrow strokeWidth=2 data={[{x: 250, y:125}, {x:250, y: 190}, {x: 10, y: 190}, {x: 10, y: 150}, {x: 65, y: 150}]} />
        <Arrow strokeWidth=2 data={[{x: 130, y:50}, {x:215, y: 90}]} />
        <Arrow strokeWidth=2 data={[{x: 130, y:150}, {x:215, y: 110}]} />
        <Arrow strokeWidth=2 data={[{x: 280, y:100}, {x:380, y: 100}]} />
  
        <!-- neurons -->
        <Block x=100 y=50 width=50 height=50 color="var(--main-color-4)" />
        <Block x=100 y=150 width=50 height=50 color="var(--main-color-4)" />
        <Block x=250 y=100 width=50 height=50 color="var(--main-color-3)" />
  
        <!-- moving data -->
        <Circle x={rnnCircle1x} y={rnnCircle1y} r=5 color="var(--main-color-1)" />
        <Circle x={rnnCircle2x} y={rnnCircle2y} r=5 color="var(--main-color-1)" />
      </g>
    </svg>
  </SvgContainer>
  <!-- usual rnn -->
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 100 150">
      <Arrow strokeWidth=1 data={[{x: 40, y:150}, {x:40, y: 100}]} />
      <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
      <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:90, y: 75}, {x:90, y: 115}, {x:60, y: 115}, {x:60, y: 100}]} />
      <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
    </svg>
  </SvgContainer>

  <!-- unfolded rnn -->
  <SvgContainer maxWidth={"800px"}>
    <svg viewBox="0 0 500 150">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text={idx+1} fontSize={12} x=50 y=140 width=15 height=15 color="var(--main-color-4)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color="var(--main-color-4)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
    </svg>
  </SvgContainer>

  <!-- multilayer unfolded rnn  -->
  <SvgContainer maxWidth={"800px"}>
    <svg viewBox="0 0 500 250">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 100)">
          <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:5}, {x:50, y: -90}]} />
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Arrow strokeWidth=1 data={[{x: 70, y:-10}, {x:140, y: -10}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block x=50 y=-10 width=30 height=30 color="var(--main-color-3)" />
          <Block text={idx+1} fontSize={12} x=50 y=140 width=15 height=15 color="var(--main-color-4)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=-80 width=25 height=15 color="var(--main-color-4)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=-20 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
    </svg>
  </SvgContainer>

  <div class="separator" />
</Container>


