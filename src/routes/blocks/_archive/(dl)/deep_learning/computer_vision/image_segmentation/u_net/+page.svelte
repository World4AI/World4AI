<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/unet.ipynb";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\unet.ipynb";

  let references = [
    {
        author: "O. Ronneberger and P.Fischer and T. Brox",
        title: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        journal: "Medical Image Computing and Computer-Assisted Intervention (MICCAI)",
        year: "2015",
        pages: "234-241",
        volume: "",
        issue: "",
    }
  ]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | U-Net</title>
  <meta
    name="description"
    content="U-net is a convolutional neural network architecture, that uses a series of contractions and expansion in order to generate segmentation masks."
  />
</svelte:head>

<h1>U-Net</h1>
<div class="separator" />

<Container>
  <p>U-Net is a convolutional neural network architecture, that was originally developed for image segmentation tasks <InternalLink type="reference" id={1} />. The network is characterized by a series of contractions, where the resolution is reduced and the number of filter maps grows and a series of expansions, where the resolution grows and the number of filter shrinks.</p>
  <p>The architecture is often depicted as a U-shaped graph. Boxes on the same level represent outputs with the same resolution, while below boxes are of lower resolution. At the lowest point the architecture reaches a so called bottleneck, after which the resolution starts to grow again.</p>
  <SvgContainer maxWidth="300px">
    <svg version="1.1" viewBox="0 0 1000 850" xmlns="http://www.w3.org/2000/svg">
     <defs>
      <marker id="TriangleStart" overflow="visible" markerHeight="10" markerWidth="8.6499996" orient="auto-start-reverse" preserveAspectRatio="xMidYMid" viewBox="0 0 5.3244081 6.1553851">
       <path transform="scale(.5)" d="m5.77 0-8.65 5v-10z" fill="context-stroke" fill-rule="evenodd" stroke="context-stroke" stroke-width="1pt"/>
      </marker>
     </defs>
     <g fill="none" stroke="#000">
      <g stroke-width="2">
       <rect x="20" y="20" width="20" height="220"/>
       <rect x="50" y="20" width="20" height="220"/>
       <rect x="80" y="20" width="20" height="220"/>
       <rect x="80" y="275" width="20" height="170"/>
       <rect x="110" y="275" width="20" height="170"/>
       <rect x="140" y="275" width="20" height="170"/>
       <rect x="139.93" y="487.69" width="20.017" height="144.63"/>
       <rect x="169.96" y="487.69" width="20.017" height="144.63"/>
       <rect x="199.98" y="487.69" width="20.017" height="144.63"/>
       <rect x="199.98" y="663.84" width="20.017" height="112.31"/>
       <rect x="230.01" y="663.84" width="20.017" height="112.31"/>
       <rect x="260.03" y="663.84" width="20.017" height="112.31"/>
       <rect x="259.92" y="820.3" width="79.781" height="19.407"/>
       <rect transform="scale(-1,1)" x="-980" y="20" width="20" height="220"/>
       <rect transform="scale(-1,1)" x="-950" y="20" width="20" height="220"/>
       <rect transform="scale(-1,1)" x="-920" y="20" width="20" height="220"/>
       <rect transform="scale(-1,1)" x="-920" y="275" width="20" height="170"/>
       <rect transform="scale(-1,1)" x="-890" y="275" width="20" height="170"/>
       <rect transform="scale(-1,1)" x="-860" y="275" width="20" height="170"/>
       <rect transform="scale(-1,1)" x="-860.07" y="487.69" width="20.017" height="144.63"/>
       <rect transform="scale(-1,1)" x="-830.04" y="487.69" width="20.017" height="144.63"/>
       <rect transform="scale(-1,1)" x="-800.02" y="487.69" width="20.017" height="144.63"/>
       <rect transform="scale(-1,1)" x="-800.02" y="663.84" width="20.017" height="112.31"/>
       <rect transform="scale(-1,1)" x="-769.99" y="663.84" width="20.017" height="112.31"/>
       <rect transform="scale(-1,1)" x="-739.97" y="663.84" width="20.017" height="112.31"/>
       <rect transform="scale(-1,1)" x="-740.08" y="820.3" width="79.781" height="19.407"/>
       <rect x="380.32" y="820.51" width="239.17" height="18.984"/>
      </g>
      <g>
       <path d="m140 120h720" marker-end="url(#TriangleStart)" stroke-dasharray="4,4" stroke-width="2"/>
       <path d="m180 360h629.91" marker-end="url(#TriangleStart)" marker-mid="url(#TriangleStart)" stroke-dasharray="3.74137998,3.74137998" stroke-width="1.8707"/>
       <path d="m225.05 560h525.94" marker-end="url(#TriangleStart)" marker-mid="url(#TriangleStart)" stroke-dasharray="3.41872001,3.41872001" stroke-width="1.7094"/>
       <path d="m297.03 720h395.55" marker-end="url(#TriangleStart)" marker-mid="url(#TriangleStart)" stroke-dasharray="2.96481991,2.96481991" stroke-width="1.4824"/>
      </g>
     </g>
    </svg>
  </SvgContainer>
  <p>Often there are skip-connections, as represented by the arrows, where the feature maps from the left and the right are concatenated, in order to produce better results.</p>
  <p>Our implementation is going to be fairly simple, we will first downsample and then upsample the image, without any skip connections. For more complex tasks skip connections will most likely be beneficieal, but will focus on the u-shaped design for the time being.</p>
</Container>
<JupyterNB {url} {fileName} />
<Footer {references} />

<style>
  rect {
    fill: var(--main-color-4);
  }
</style>
