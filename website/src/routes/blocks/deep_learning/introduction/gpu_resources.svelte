<script>
  import Container from "$lib/Container.svelte";
  import GraphicsCard from "./_gpu_resources/GraphicsCard.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | GPU Resources</title>
  <meta
    name="description"
    content="It is essential to have access to a modern GPU if you are interested in working in the field of deep learning. For students free alternatives like google colab should be sufficient."
  />
</svelte:head>

<h1>GPU Resources</h1>
<div class="separator" />
<Container>
  <SvgContainer maxWidth="500px">
    <GraphicsCard />
  </SvgContainer>
  <p>
    Deep learning is extremely data and computationally hungry. You require
    modern Nvidia grapics cards for training and inference, but that option
    might not be available to everyone. If you have your own workstation with a
    number of GPUs, then we congratulate you on that achievement. If you are at
    the beginning of your carreer or live in a part of the world where it is
    unfeasable to afford your own computational resources, there are either
    cheap or free alternatives.
  </p>
  <div class="separator" />

  <h2>Free Cloud Resources</h2>

  <p>
    <svg viewBox="0 0 24 24" focusable="false" height="30px"
      ><g>
        <path
          d="M4.54,9.46,2.19,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36A3.59,3.59,0,0,1,4.54,9.46Z"
          style=""
          fill="#E8710A"
        />
        <path
          d="M2.19,7.1,4.54,9.46a3.59,3.59,0,0,1,5.08,0l1.71-2.93h0l-.1-.08h0A6.93,6.93,0,0,0,2.19,7.1Z"
          style=""
          fill="#F9AB00"
        />
        <path
          d="M11.34,17.46h0L9.62,14.54a3.59,3.59,0,0,1-5.08,0L2.19,16.9a6.93,6.93,0,0,0,9,.65l.11-.09"
          style=""
          fill="#F9AB00"
        />
        <path
          d="M12,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36a3.59,3.59,0,1,1,5.08-5.08L21.81,7.1A6.93,6.93,0,0,0,12,7.1Z"
          style=""
          fill="#F9AB00"
        />
        <path
          d="M21.81,7.1,19.46,9.46a3.59,3.59,0,0,1-5.08,5.08L12,16.9A6.93,6.93,0,0,0,21.81,7.1Z"
          style=""
          fill="#E8710A"
        />
      </g></svg
    >
    <a href="https://colab.research.google.com" target="_blank">Google Colab</a>
    (Colaboratory) provides you with free cloud compute in the form of Jupyter Notebooks.
    You can switch between CPU, GPU and TPU resources. If you use the platform extensively
    without a break they might switch you to a lower tier server or give you a time
    out, but considering the cost is zero, this is a great deal. Google also provides
    payed tiers with better GPUs and lower chance of getting disconnected. For learning
    purposes the free tier should be sufficient.
  </p>

  <p>
    <a href="https://www.kaggle.com" target="_blank">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 163 63.2"
        height="30px"
      >
        <defs>
          <style>
            .cls-1 {
              fill: #20beff;
            }
          </style>
        </defs>
        <g id="Layer_2" data-name="Layer 2">
          <g id="Layer_1-2" data-name="Layer 1">
            <path
              class="cls-1"
              d="M26.92 47c-.05.18-.24.27-.56.27h-6.17a1.24 1.24 0 0 1-1-.48L9 33.78l-2.83 2.71v10.06a.61.61 0 0 1-.69.69H.69a.61.61 0 0 1-.69-.69V.69A.61.61 0 0 1 .69 0h4.79a.61.61 0 0 1 .69.69v28.24l12.21-12.35a1.44 1.44 0 0 1 1-.49h6.39a.54.54 0 0 1 .55.35.59.59 0 0 1-.07.63L13.32 29.55l13.46 16.72a.65.65 0 0 1 .14.73ZM51.93 47.24h-4.79c-.51 0-.76-.23-.76-.69v-1a12.77 12.77 0 0 1-7.84 2.29A11.28 11.28 0 0 1 31 45.16a9 9 0 0 1-3.12-7.07q0-6.81 8.46-9.23a61.55 61.55 0 0 1 10.06-1.67A5.47 5.47 0 0 0 40.48 21a14 14 0 0 0-7.91 2.77c-.41.24-.71.19-.9-.13l-2.5-3.54c-.23-.28-.16-.6.21-1a19.32 19.32 0 0 1 11.1-3.68A13.29 13.29 0 0 1 48 17.55q4.59 3.06 4.58 9.78v19.22a.61.61 0 0 1-.65.69Zm-5.55-14.5q-6.8.7-9.3 1.81Q33.69 36 34 38.71a3.49 3.49 0 0 0 1.53 2.46 5.87 5.87 0 0 0 3 1.08 9.49 9.49 0 0 0 7.77-2.57ZM81 59.28q-3.81 3.92-10.74 3.92a15.41 15.41 0 0 1-7.63-2c-.51-.33-1.11-.76-1.81-1.29s-1.5-1.19-2.43-2a.72.72 0 0 1-.07-1l3.26-3.26a.76.76 0 0 1 .56-.21.68.68 0 0 1 .49.21c2.58 2.58 5.11 3.88 7.56 3.88q8.39 0 8.39-8.74v-3.63a13.1 13.1 0 0 1-8.67 2.71 12.48 12.48 0 0 1-10.55-5.07A18.16 18.16 0 0 1 56 31.63a18 18 0 0 1 3.2-10.82 12.19 12.19 0 0 1 10.61-5.34 13.93 13.93 0 0 1 8.74 2.71v-1.39a.62.62 0 0 1 .69-.7h4.79a.62.62 0 0 1 .7.7v31q.03 7.57-3.73 11.49ZM78.58 26q-1.74-4.44-8-4.44-8.11 0-8.11 10.12 0 5.63 2.7 8.19a7.05 7.05 0 0 0 5.21 2q6.51 0 8.25-4.44ZM113.59 59.28q-3.78 3.91-10.72 3.92a15.44 15.44 0 0 1-7.63-2q-.76-.49-1.8-1.29c-.7-.53-1.51-1.19-2.43-2a.7.7 0 0 1-.07-1l3.26-3.26a.74.74 0 0 1 .55-.21.67.67 0 0 1 .49.21c2.59 2.58 5.11 3.88 7.56 3.88q8.4 0 8.4-8.74v-3.63a13.14 13.14 0 0 1-8.68 2.71A12.46 12.46 0 0 1 92 42.8a18.09 18.09 0 0 1-3.33-11.17 18 18 0 0 1 3.19-10.82 12.21 12.21 0 0 1 10.61-5.34 14 14 0 0 1 8.75 2.71v-1.39a.62.62 0 0 1 .69-.7h4.79a.62.62 0 0 1 .69.7v31q-.02 7.57-3.8 11.49ZM111.2 26q-1.74-4.44-8-4.44-8.2-.05-8.2 10.07 0 5.63 2.71 8.19a7 7 0 0 0 5.2 2q6.53 0 8.26-4.44ZM128 47.24h-4.78a.62.62 0 0 1-.7-.69V.69a.62.62 0 0 1 .7-.69H128a.61.61 0 0 1 .7.69v45.86a.61.61 0 0 1-.7.69ZM162.91 33.16a.62.62 0 0 1-.7.69h-22.54a8.87 8.87 0 0 0 2.91 5.69 10.63 10.63 0 0 0 7.15 2.46 11.64 11.64 0 0 0 6.86-2.15c.42-.28.77-.28 1 0l3.26 3.33c.37.37.37.69 0 1a18.76 18.76 0 0 1-11.58 3.75 16 16 0 0 1-11.8-4.72 16.2 16.2 0 0 1-4.57-11.86 16 16 0 0 1 4.51-11.52 14.36 14.36 0 0 1 10.82-4.3A14.07 14.07 0 0 1 158.88 20 15 15 0 0 1 163 31.63ZM153.82 23a8.18 8.18 0 0 0-5.69-2.15 8.06 8.06 0 0 0-5.48 2.08 9.24 9.24 0 0 0-3 5.41h16.71a7 7 0 0 0-2.54-5.34Z"
            />
          </g>
        </g>
      </svg>
    </a>

    is a plattform designed to run machine leanring competitions. For that
    purpose they provide you weekly resources with at least 30 GPU hours and at
    least 30 TPU hours. Unlike Google Colab, Kaggle will not disconnect you, but
    when the hours run out you have to wait until the resources are reset. Kolab
    was acquired by Google, therefore the computational resources are similar.
  </p>

  <p>
    If you are looking for alternatives free of Google,
    <a href="https://www.paperspace.com" target="_blank">Paperspace</a>
    might be it. Similar to Google Colab and Kaggle, Paperspace provides free cloud
    GPU notebooks.
  </p>
  <div class="separator" />

  <h2>Cheap Cloud Resources</h2>
  <p>
    There are tons of cloud providers that can hook you up with payed GPU
    instances. AWS by Amazon, Azure by Microsoft and Google Cloud are the top
    three players. Below we have compiled a couple of smaller alternatives, that
    should provide you with a more budget friedly experience.
  </p>
  <p>
    <a href="https://lambdalabs.com" target="_blank">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 569.03 131.61"
        height="30px"
        ><path
          d="M181.51,90.48h31.91v15.24H163.3V25.48h18.21ZM4.87,5.62H126.81V126H4.87ZM15.65,115.39H116V16.22H15.65ZM37.08,25.64,58.29,62.82l-23.6,43.06h15.2L65.62,76.49l16.75,29.39H97.85L52.56,25.63ZM348.81,46.2a21,21,0,0,0-11.07,2.86,25.84,25.84,0,0,0-7.5,7H330c-2.5-6.07-8.45-9.88-15.95-9.88-10,0-15.12,5.47-17.62,9.17h-.36l-.83-7.86H280.48v58.21H298.1V71.79a11.91,11.91,0,0,1,.71-4.52c1.07-3,3.81-6.55,8.45-6.55,5.72,0,8.34,5,8.34,12.38v32.62h17.62v-34a15.06,15.06,0,0,1,.71-4.65,8.86,8.86,0,0,1,8.33-6.31c5.84,0,8.58,4.88,8.58,13.34v31.66h17.61V71.44C368.45,54.06,360,46.2,348.81,46.2Zm136.62-23h18.1l0,82.47H487.93l-.83-8.57h-.24c-3.69,6.55-10.72,9.88-18.22,9.88-13.81,0-24.88-11.78-24.88-29.88-.12-19.64,12.14-30.95,26.07-30.95,7.15,0,12.74,2.5,15.36,6.55h.24Zm0,48.9a28.89,28.89,0,0,0-.24-3.21c-1.07-4.88-5-8.93-10.59-8.93-8.22,0-12.5,7.38-12.5,16.55,0,9.88,4.88,16.07,12.38,16.07A10.58,10.58,0,0,0,485,83.94a15.3,15.3,0,0,0,.48-4.17Zm78.08-.84v34.41h-15.6l-1.07-5.83h-.36c-3.82,4.64-9.8,7.14-16.73,7.14-11.83,0-18.88-8.57-18.88-17.86,0-15.12,15.39-19.56,34.29-19.56V66.07c0-3.09-1.67-7.5-10.63-7.5a33.85,33.85,0,0,0-13.67,3.09l-5.8-10.35c4.06-2.26,12.06-5.12,22.7-5.12C557.23,46.19,563.51,57.62,563.51,71.31Zm-17.75,9c-10,0-18.11.68-18.15,8,0,4.67,4.29,5.93,8.35,5.93a9.91,9.91,0,0,0,9.44-6.67,12.83,12.83,0,0,0,.36-3.09Zm-109-4.76c0,20.94-13.16,31.4-26.32,31.4-6.7,0-13.16-2.5-17.27-9.39H393l-.7,8.08H377.6V23.25h17.84V54.38h.23c3.41-5,9.4-8.2,17.39-8.2C426.81,46.18,436.91,57.71,436.8,75.55Zm-18.22.72c0-9-4-16.18-12.33-16.18-4.82,0-9.28,3.69-10.46,8.92a15.53,15.53,0,0,0-.35,3.33v8.57a13.72,13.72,0,0,0,.35,3.21,10.85,10.85,0,0,0,10.46,8.56C413.89,92.68,418.58,86.73,418.58,76.27Zm-146-5v34.41H257l-1.07-5.83h-.36c-3.83,4.64-9.8,7.14-16.73,7.14C227,107,220,98.46,220,89.17c0-15.12,15.39-19.56,34.29-19.56V66.07c0-3.09-1.67-7.5-10.63-7.5A33.77,33.77,0,0,0,230,61.66l-5.81-10.35c4.07-2.26,12.07-5.12,22.71-5.12C266.33,46.19,272.61,57.62,272.61,71.31Zm-17.75,9c-10,0-18.11.68-18.15,8,0,4.67,4.29,5.93,8.35,5.93a9.9,9.9,0,0,0,9.44-6.67,12.83,12.83,0,0,0,.36-3.09Z"
        /></svg
      >
    </a> is specialized in GPU compute and can provide cloud services. Additionally
    the offer workstations and notebooks developed for deep learning.
  </p>

  <p>
    <a href="https://www.linode.com" target="_blank">
      <svg
        viewBox="0 0 506.94 131.5"
        height="30px"
        xmlns="http://www.w3.org/2000/svg"
        ><path
          d="m109.65 58.21-18.21-10.07-15.37 9.38-.19 9.59-7.47-4.92-10.16 6.19-.44-10.48-10.48-7 10-5.17c-.08 0 0 1-1.48-34.34l-23.65-11.39-32.2 10 7.19 34.63 10.81 8.37-8.22 3.9 5.37 26.1 7.55 7.09-5.4 3.29 4.19 20.18 16.94 17.94c.08-.1 1.78-1.41 21.79-17.32l-.58-13.71 8.58 7.28c.12-.12 1.84-1.39 17.56-13.89l.61-10.14 6.48 4.5c.13-.12 1.58-1.22 14.26-11.33z"
          fill="#231f20"
        /><path d="m94.54 69 15.11-10.79-18.21-10.07-15.37 9.38z" /><path
          d="m92.87 88.2 1.67-19.2-18.47-11.48-.38 18.75z"
          fill="#004b16"
        /><path
          d="m68.22 107.73v-19.89l-18.66-14.15 1.47 19.54z"
          fill="#004b16"
        /><path d="m68.22 87.84 18.74-13.43-18.55-12.22-18.85 11.5z" /><path
          d="m38.43 131.48-2.98-20.32-18.15-17.8 4.19 20.18z"
          fill="#004b16"
        /><path d="m35.45 111.16 23.91-17.06-18.53-15.09-23.53 14.35z" /><path
          d="m33.9 100.6-3.94-26.88-20.22-16.81 5.41 26.07z"
          fill="#004b16"
        /><path d="m29.96 73.72 27.85-15.82-20.8-13.94-27.27 12.95z" /><path
          d="m28.07 60.88-5.4-36.81-22.67-14.06 7.19 34.62z"
          fill="#004b16"
        /><path d="m22.67 24.07 33.14-12.7-23.61-11.37-32.2 10.01z" /><g
          fill="#1cb35c"
          ><path
            d="m107.13 76.87c-14.02 11.13-14.33 11.33-14.26 11.33 1.74-20.1 1.59-19.2 1.67-19.2 16-11.45 15-10.79 15.11-10.79z"
          /><path
            d="m85.78 93.84c-17.35 13.8-17.63 13.89-17.56 13.89-.17-20.82-.07-19.89 0-19.89 20-14.3 18.67-13.43 18.74-13.43z"
          /><path
            d="m60.22 114.16c-21.66 17.22-21.86 17.32-21.79 17.32-3.07-20.94-3-20.32-3-20.32 25.47-18.16 23.86-17.06 23.93-17.06z"
          /><path
            d="m55.81 11.37c1.52 35.37 1.4 34.34 1.48 34.34-28.66 14.89-29.29 15.17-29.22 15.17-5.52-37.63-5.47-36.81-5.4-36.81z"
          /><path
            d="m57.81 57.9c1.15 26.81 1 25.88 1.11 25.88-24.81 16.67-25.09 16.82-25 16.82-4-27.58-4-26.88-3.94-26.88z"
          /></g
        ><path
          d="m151.61 14.24 16.58-4v79.88q0 13.13 7.83 15.65-3.84 7.31-13.13 7.3-11.28 0-11.28-15.66z"
        /><path
          d="m186.66 111.72v-57.42h-9.08v-13.6h25.86v71zm8.56-98.54a9.62 9.62 0 1 1 -9.62 9.62 9.63 9.63 0 0 1 9.62-9.62z"
        /><path
          d="m262.94 111.74v-41.06c0-6.05-1.16-10.48-3.48-13.26s-6.11-4.18-11.37-4.18a17.74 17.74 0 0 0 -7.8 2.06 18 18 0 0 0 -6.46 5.1v51.34h-16.59v-71h11.94l3 6.64q6.76-8 20-8 12.66 0 20 7.59t7.33 21.19v43.58z"
        /><path
          d="m288.42 76.06q0-16.26 9.38-26.47t24.77-10.21q16.19 0 25.14 9.82t9 26.86q0 17-9.12 27t-25 10q-16.18 0-25.17-10.12t-9-26.88zm17.24 0q0 23.47 16.91 23.48a14.54 14.54 0 0 0 12.31-6.11q4.55-6.09 4.54-17.37 0-23.14-16.85-23.15a14.61 14.61 0 0 0 -12.33 6.09q-4.58 6.11-4.58 17.06z"
        /><path
          d="m412.37 111.74v-4.31c-1.37 1.5-3.71 2.82-7 3.95a31.33 31.33 0 0 1 -10.15 1.69q-14.87 0-23.38-9.42t-8.52-26.27q0-16.84 9.78-27.42a32 32 0 0 1 24.51-10.58 32.37 32.37 0 0 1 14.72 3.32v-28.46l16.58-4v101.5zm0-54.06a17.6 17.6 0 0 0 -11.07-4.24q-10 0-15.33 6.07t-5.37 17.41q0 22.15 21.36 22.15a16.11 16.11 0 0 0 5.87-1.42c2.32-1 3.83-1.92 4.54-2.89z"
        /><path
          d="m505.55 81.3h-50.74q.46 8.49 5.83 13.2t14.46 4.7q11.34 0 17.25-5.9l6.43 12.7q-8.75 7.09-26.13 7.1-16.25 0-25.7-9.52t-9.45-26.58q0-16.78 10.38-27.2a33.91 33.91 0 0 1 24.9-10.41q15.45 0 24.81 9.22t9.35 23.48a46.28 46.28 0 0 1 -1.39 9.21zm-50.15-12.47h34.89q-1.73-15.58-17.24-15.59-14.2 0-17.65 15.59z"
        /></svg
      >
    </a> offers different cloud solutions, including GPU servers.
  </p>
  <div class="separator" />
</Container>
