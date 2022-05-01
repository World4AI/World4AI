<script>
  import Question from "$lib/Question.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | GPU Resources</title>
  <meta
    name="description"
    content="It is essential to have access to a modern GPU if you are interested in working in the field of deep learning. For students free alternatives like google colab should be sufficient."
  />
</svelte:head>

<h1>GPU Resources</h1>
<Question>What options do you have to train your model on the GPU?</Question>
<div class="separator" />
<p>
  Deep learning is extremely data and computationally hungry. You require modern
  Nvidia grapics cards for training and inference, but that option might not be
  available to everyone. If you have your own workstation with a number of GPUs,
  then we congratulate you on that achievement. If you are at the beginning of
  your carreer or live in a part of the world where it is unfeasable to afford
  your own computational resources, there are either cheap or free alternatives.
</p>
<div class="separator" />

<h2>Free Cloud Resources</h2>

<h3>
  <svg viewBox="0 0 24 24" width="40px"
    ><g fill="var(--text-color)">
      <path
        d="M4.54,9.46,2.19,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36A3.59,3.59,0,0,1,4.54,9.46Z"
      />
      <path
        d="M2.19,7.1,4.54,9.46a3.59,3.59,0,0,1,5.08,0l1.71-2.93h0l-.1-.08h0A6.93,6.93,0,0,0,2.19,7.1Z"
      />
      <path
        d="M11.34,17.46h0L9.62,14.54a3.59,3.59,0,0,1-5.08,0L2.19,16.9a6.93,6.93,0,0,0,9,.65l.11-.09"
      />
      <path
        d="M12,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36a3.59,3.59,0,1,1,5.08-5.08L21.81,7.1A6.93,6.93,0,0,0,12,7.1Z"
      />
      <path
        d="M21.81,7.1,19.46,9.46a3.59,3.59,0,0,1-5.08,5.08L12,16.9A6.93,6.93,0,0,0,21.81,7.1Z"
      />
    </g></svg
  >
  Google Colab
</h3>
<p>
  <a href="https://colab.research.google.com" target="_blank">Google Colab</a> (Colaboratory)
  provides you with free cloud compute in the form of Jupyter Notebooks. You can
  switch between CPU, GPU and TPU resources. If you use the platform extensively
  without a break they might switch you to a lower tier server or give you a time
  out, but considering the cost is zero, this is a great deal. Google also provides
  payed tiers with better GPUs and lower chance of getting disconnected. For learning
  purposes the free tier should be sufficient.
</p>
<div class="separator" />

<h3>
  <svg width="50px" fill="var(--text-color)" viewBox="0 0 163 63.2">
    <g>
      <g>
        <path
          d="M26.92 47c-.05.18-.24.27-.56.27h-6.17a1.24 1.24 0 0 1-1-.48L9 33.78l-2.83 2.71v10.06a.61.61 0 0 1-.69.69H.69a.61.61 0 0 1-.69-.69V.69A.61.61 0 0 1 .69 0h4.79a.61.61 0 0 1 .69.69v28.24l12.21-12.35a1.44 1.44 0 0 1 1-.49h6.39a.54.54 0 0 1 .55.35.59.59 0 0 1-.07.63L13.32 29.55l13.46 16.72a.65.65 0 0 1 .14.73ZM51.93 47.24h-4.79c-.51 0-.76-.23-.76-.69v-1a12.77 12.77 0 0 1-7.84 2.29A11.28 11.28 0 0 1 31 45.16a9 9 0 0 1-3.12-7.07q0-6.81 8.46-9.23a61.55 61.55 0 0 1 10.06-1.67A5.47 5.47 0 0 0 40.48 21a14 14 0 0 0-7.91 2.77c-.41.24-.71.19-.9-.13l-2.5-3.54c-.23-.28-.16-.6.21-1a19.32 19.32 0 0 1 11.1-3.68A13.29 13.29 0 0 1 48 17.55q4.59 3.06 4.58 9.78v19.22a.61.61 0 0 1-.65.69Zm-5.55-14.5q-6.8.7-9.3 1.81Q33.69 36 34 38.71a3.49 3.49 0 0 0 1.53 2.46 5.87 5.87 0 0 0 3 1.08 9.49 9.49 0 0 0 7.77-2.57ZM81 59.28q-3.81 3.92-10.74 3.92a15.41 15.41 0 0 1-7.63-2c-.51-.33-1.11-.76-1.81-1.29s-1.5-1.19-2.43-2a.72.72 0 0 1-.07-1l3.26-3.26a.76.76 0 0 1 .56-.21.68.68 0 0 1 .49.21c2.58 2.58 5.11 3.88 7.56 3.88q8.39 0 8.39-8.74v-3.63a13.1 13.1 0 0 1-8.67 2.71 12.48 12.48 0 0 1-10.55-5.07A18.16 18.16 0 0 1 56 31.63a18 18 0 0 1 3.2-10.82 12.19 12.19 0 0 1 10.61-5.34 13.93 13.93 0 0 1 8.74 2.71v-1.39a.62.62 0 0 1 .69-.7h4.79a.62.62 0 0 1 .7.7v31q.03 7.57-3.73 11.49ZM78.58 26q-1.74-4.44-8-4.44-8.11 0-8.11 10.12 0 5.63 2.7 8.19a7.05 7.05 0 0 0 5.21 2q6.51 0 8.25-4.44ZM113.59 59.28q-3.78 3.91-10.72 3.92a15.44 15.44 0 0 1-7.63-2q-.76-.49-1.8-1.29c-.7-.53-1.51-1.19-2.43-2a.7.7 0 0 1-.07-1l3.26-3.26a.74.74 0 0 1 .55-.21.67.67 0 0 1 .49.21c2.59 2.58 5.11 3.88 7.56 3.88q8.4 0 8.4-8.74v-3.63a13.14 13.14 0 0 1-8.68 2.71A12.46 12.46 0 0 1 92 42.8a18.09 18.09 0 0 1-3.33-11.17 18 18 0 0 1 3.19-10.82 12.21 12.21 0 0 1 10.61-5.34 14 14 0 0 1 8.75 2.71v-1.39a.62.62 0 0 1 .69-.7h4.79a.62.62 0 0 1 .69.7v31q-.02 7.57-3.8 11.49ZM111.2 26q-1.74-4.44-8-4.44-8.2-.05-8.2 10.07 0 5.63 2.71 8.19a7 7 0 0 0 5.2 2q6.53 0 8.26-4.44ZM128 47.24h-4.78a.62.62 0 0 1-.7-.69V.69a.62.62 0 0 1 .7-.69H128a.61.61 0 0 1 .7.69v45.86a.61.61 0 0 1-.7.69ZM162.91 33.16a.62.62 0 0 1-.7.69h-22.54a8.87 8.87 0 0 0 2.91 5.69 10.63 10.63 0 0 0 7.15 2.46 11.64 11.64 0 0 0 6.86-2.15c.42-.28.77-.28 1 0l3.26 3.33c.37.37.37.69 0 1a18.76 18.76 0 0 1-11.58 3.75 16 16 0 0 1-11.8-4.72 16.2 16.2 0 0 1-4.57-11.86 16 16 0 0 1 4.51-11.52 14.36 14.36 0 0 1 10.82-4.3A14.07 14.07 0 0 1 158.88 20 15 15 0 0 1 163 31.63ZM153.82 23a8.18 8.18 0 0 0-5.69-2.15 8.06 8.06 0 0 0-5.48 2.08 9.24 9.24 0 0 0-3 5.41h16.71a7 7 0 0 0-2.54-5.34Z"
        />
      </g>
    </g>
  </svg>
  Kaggle
</h3>
<p>
  <a href="https://www.kaggle.com" target="_blank">Kaggle</a> is a plattform designed
  to run machine leanring competitions. For that purpose they provide you weekly
  resources with at least 30 GPU hours and at least 30 TPU hours. Unlike Google Colab,
  Kaggle will not disconnect you, but when the hours run out you have to wait until
  the resources are reset. Kolab was acquired by Google, therefore the computational
  resources are similar.
</p>
<div class="separator" />

<h3>Paperspace</h3>
<p>
  If you are looking for alternatives free of Google, <a
    href="https://www.paperspace.com"
    target="_blank">Paperspace</a
  > might be it. Similar to Google Colab and Kaggle, Paperspace provides free cloud
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
<h3>
  <svg version="1.1" width="40px" viewBox="0 0 131.61 131.61">
    <path
      fill="var(--text-color)"
      d="m4.87 5.62h121.94v120.38h-121.94zm10.78 109.77h100.35v-99.17h-100.35zm21.43-89.75 21.21 37.18-23.6 43.06h15.2l15.73-29.39 16.75 29.39h15.48l-45.29-80.25z"
    />
  </svg>
  Lambda
</h3>
<p>
  <a href="https://lambdalabs.com" target="_blank">Lambda</a> is specialized in GPU
  compute and can provide cloud services. Additionally the offer workstations and
  notebooks developed for deep learning.
</p>
<div class="separator" />

<h3>
  <svg viewBox="0 0 110 131.5" width="40px" xmlns="http://www.w3.org/2000/svg">
    <g stroke="var(--text-color)" fill="none">
      <path
        d="m109.65 58.21-18.21-10.07-15.37 9.38-.19 9.59-7.47-4.92-10.16 6.19-.44-10.48-10.48-7 10-5.17c-.08 0 0 1-1.48-34.34l-23.65-11.39-32.2 10 7.19 34.63 10.81 8.37-8.22 3.9 5.37 26.1 7.55 7.09-5.4 3.29 4.19 20.18 16.94 17.94c.08-.1 1.78-1.41 21.79-17.32l-.58-13.71 8.58 7.28c.12-.12 1.84-1.39 17.56-13.89l.61-10.14 6.48 4.5c.13-.12 1.58-1.22 14.26-11.33z"
      /><path d="m94.54 69 15.11-10.79-18.21-10.07-15.37 9.38z" /><path
        d="m92.87 88.2 1.67-19.2-18.47-11.48-.38 18.75z"
      /><path d="m68.22 107.73v-19.89l-18.66-14.15 1.47 19.54z" /><path
        d="m68.22 87.84 18.74-13.43-18.55-12.22-18.85 11.5z"
      /><path d="m38.43 131.48-2.98-20.32-18.15-17.8 4.19 20.18z" /><path
        d="m35.45 111.16 23.91-17.06-18.53-15.09-23.53 14.35z"
      /><path d="m33.9 100.6-3.94-26.88-20.22-16.81 5.41 26.07z" /><path
        d="m29.96 73.72 27.85-15.82-20.8-13.94-27.27 12.95z"
      /><path d="m28.07 60.88-5.4-36.81-22.67-14.06 7.19 34.62z" /><path
        d="m22.67 24.07 33.14-12.7-23.61-11.37-32.2 10.01z"
      /><g
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
      ></g
    >
  </svg>
  Linode
</h3>
<p>
  <a href="https://www.linode.com" target="_blank">Linode</a> offers different cloud
  solutions, including GPU servers.
</p>
<div class="separator" />
