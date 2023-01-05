<script>
  import Container from "$lib/Container.svelte";
  import Alert from "$lib/Alert.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import GraphicsCard from "../_gpu_resources/GraphicsCard.svelte";
  import Blackboard from "../_education/Blackboard.svelte";
</script>

<svelte:head>
  <title />
  <meta name="description" content="" />
</svelte:head>

<h1>Deep Learning Frameworks</h1>
<div class="separator" />
<Container>
  <p>
    Deep learning algorithms are usually implemented with specialized deep
    learning frameworks. These frameworks can run on the GPU to improve the
    speed of training and execution, have build in functionality that is
    commonly used in deep learning research and are battle tested. Implementing
    all those details by hand is usually a bad idea, unless you do it for
    learning purposes.
  </p>
  <p>
    Deep learning frameworks have a rich history. If we wrote this chapter two
    years ago the content would look different. Five years ago the contents
    would be different again. That says a lot about the speed with which the
    development of those frameworks progresses.
  </p>
  <p>
    All of the below mentioned frameworks use Python as the interface language
    between the programmer and the underlying execution. The frameworks
    themselves on the other hand are written in C++ and CUDA to optimize
    performance. There are fromeworks that use other programming languages
    instead of Python like R, Julia or even JavaScript, but mostly research and
    education is done in Python so you are out of luck if you prefer any other
    programming language. Python is still king in the deep learning community.
  </p>
  <div class="separator" />

  <h2>
    TensorFlow
    <svg
      xmlns="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink"
      width="32px"
      viewBox="0 0 54 64"
      ><g
        ><g
          ><defs
            ><path
              id="SVGID_1_"
              d="M47.5 17.6L25 4.8v52.6l9-5.2V37.4l6.8 3.9-.1-10.1-6.7-3.9v-5.9l13.5 7.9z"
            /></defs
          ><clipPath id="SVGID_2_"
            ><use xlink:href="#SVGID_1_" overflow="visible" /></clipPath
          ><g clip-path="url(#SVGID_2_)"
            ><linearGradient
              id="SVGID_3_"
              gradientUnits="userSpaceOnUse"
              x1="-1.6"
              y1="335.05"
              x2="53.6"
              y2="335.05"
              gradientTransform="translate(0 -304)"
              ><stop offset="0" stop-color="#ff6f00" /><stop
                offset="1"
                stop-color="#ffa800"
              /></linearGradient
            ><path d="M-1.6 4.6h55.2v52.9H-1.6V4.6z" fill="url(#SVGID_3_)" /></g
          ></g
        ></g
      ><g
        ><g
          ><defs
            ><path
              id="SVGID_4_"
              d="M.5 17.6L23 4.8v52.6l-9-5.2V21.4L.5 29.3z"
            /></defs
          ><clipPath id="SVGID_5_"
            ><use xlink:href="#SVGID_4_" overflow="visible" /></clipPath
          ><g clip-path="url(#SVGID_5_)"
            ><linearGradient
              id="SVGID_6_"
              gradientUnits="userSpaceOnUse"
              x1="-1.9"
              y1="335.05"
              x2="53.3"
              y2="335.05"
              gradientTransform="translate(0 -304)"
              ><stop offset="0" stop-color="#ff6f00" /><stop
                offset="1"
                stop-color="#ffa800"
              /></linearGradient
            ><path d="M-1.9 4.6h55.2v52.9H-1.9V4.6z" fill="url(#SVGID_6_)" /></g
          ></g
        ></g
      >
    </svg>
    and Keras
    <!-- https://commons.wikimedia.org/wiki/File:Keras_logo.svg -->
    <svg
      version="1.1"
      id="Layer_1"
      xmlns="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink"
      width="32px"
      height="32px"
      x="0px"
      y="0px"
      viewBox="0 0 1080 1080"
      style="enable-background:new 0 0 1080 1080;"
      xml:space="preserve"
    >
      <style type="text/css">
        .st0 {
          fill: #ffffff;
        }
        .st1 {
          fill: #d00000;
        }
      </style>
      <g>
        <path
          class="st0"
          d="M1080,1079.96c0,0.02-0.02,0.04-0.04,0.04H0.04c-0.02,0-0.04-0.02-0.04-0.04V0.04C0,0.02,0.02,0,0.04,0
		h1079.93c0.02,0,0.04,0.02,0.04,0.04V1079.96z"
        />
      </g>
      <g>
        <path
          class="st1"
          d="M1062,1061.96c0,0.02-0.02,0.04-0.04,0.04H18.04c-0.02,0-0.04-0.02-0.04-0.04V18.04
		c0-0.02,0.02-0.04,0.04-0.04h1043.93c0.02,0,0.04,0.02,0.04,0.04V1061.96z"
        />
      </g>
      <g>
        <path
          class="st0"
          d="M303,823.67c0,0.79,0.46,1.89,1.01,2.44l17.87,17.87c0.56,0.56,1.66,1.01,2.44,1.01h61.15
		c0.79,0,1.89-0.46,2.44-1.01l17.87-17.87c0.56-0.56,1.01-1.66,1.01-2.44V629.64c0-0.79,0.47-1.88,1.04-2.42l77.69-74.2
		c0.57-0.54,1.4-0.46,1.84,0.2l196.29,290.6c0.44,0.65,1.45,1.19,2.23,1.19h86.63c0.79,0,1.73-0.57,2.09-1.27l15.72-30.46
		c0.36-0.7,0.29-1.8-0.16-2.45L560.56,478.03c-0.45-0.65-0.36-1.63,0.2-2.19l211.18-210.19c0.56-0.56,1.01-1.65,1.01-2.44v-3.88
		c0-0.79-0.26-2.02-0.57-2.75l-12.18-28.01c-0.31-0.72-1.22-1.31-2-1.31h-85.63c-0.79,0-1.89,0.46-2.44,1.01l-262.31,263.3
		c-0.56,0.56-1.01,0.37-1.01-0.42V249.6c0-0.79-0.44-1.9-0.98-2.48l-17.53-18.8c-0.54-0.58-1.62-1.05-2.41-1.05h-61.57
		c-0.79,0-1.87,0.47-2.41,1.05l-17.95,19.38c-0.54,0.58-0.97,1.69-0.97,2.48V823.67z"
        />
      </g>
    </svg>
  </h2>
  <p>
    <a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"
      >TensorFlow</a
    >
    is Google's flagship deep learning framework. The framework was first released
    in the year 2015, which makes it the oldest in the list.
    <a href="https://keras.io/" target="_blank" rel="noreferrer">Keras</a> is a higher
    level api that sits on top of TensorFlow and provides a lot of convenience for
    developers by reducing a lot of overhead. Keras was actually developed independently
    by François Chollet in order to facilitate fast and easy deep learning development.
    Originally Keras supported multiple deep learning libraries (for example Theano),
    but became an integral part of TensorFlow 2 in 2019 and nowadays only supports
    TensorFlow. You can say that TensorFlow and Keras go hand in hand. Keras is an
    incredible great choice for beginners. It has a vibrant community and you will
    have no difficulty to find tutorials and books to learn from.
  </p>
  <div class="separator" />

  <h2>
    PyTorch
    <svg
      version="1.1"
      id="Layer_1"
      xmlns="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink"
      x="0px"
      y="0px"
      width="160px"
      viewBox="0.6 539.9 487.3 120.2"
      enable-background="new 0.6 539.9 487.3 120.2"
      xml:space="preserve"
    >
      <g>
        <path
          fill="#EE4C2C"
          d="M63.1,567.3l-6.6,6.6c10.8,10.8,10.8,28.2,0,38.8c-10.8,10.8-28.2,10.8-38.8,0c-10.8-10.8-10.8-28.2,0-38.8
		l0,0l17.1-17.1l2.4-2.4l0,0v-12.9l-25.8,25.8c-14.4,14.4-14.4,37.6,0,52s37.6,14.4,51.7,0C77.5,604.8,77.5,581.7,63.1,567.3z"
        />
        <circle fill="#EE4C2C" cx="50.2" cy="560.9" r="4.8" />
      </g>
      <g>
        <g>
          <path
            fill="#FFFFFF"
            d="M129.8,600.3h-11.1v28.5h-8.4v-81.1c0,0,19.2,0,20.4,0c21.3,0,31.5,10.5,31.5,25.2
			C162.5,591,149.9,600.3,129.8,600.3z M130.7,555.8c-0.9,0-11.7,0-11.7,0v37.3l11.4-0.3c15.3-0.3,23.7-6.3,23.7-18.9
			C154.1,562.1,145.7,555.8,130.7,555.8z"
          />
          <path
            fill="#FFFFFF"
            d="M199.8,628.5l-4.8,12.9c-5.4,14.4-11.1,18.6-19.2,18.6c-4.5,0-7.8-1.2-11.4-2.7l2.4-7.5
			c2.7,1.5,5.7,2.7,9,2.7c4.5,0,7.8-2.4,12.3-13.8l3.9-10.5l-23.1-58.6h8.7l18.6,49l18.3-49h8.4L199.8,628.5z"
          />
          <path
            fill="#FFFFFF"
            d="M250.3,555.8v73.3h-8.4v-73.3h-28.5V548h65.2v7.8C278.5,555.8,250.3,555.8,250.3,555.8z"
          />
          <path
            fill="#FFFFFF"
            d="M302.3,630.6c-16.5,0-28.5-12.3-28.5-31.2c0-18.9,12.6-31.5,29.4-31.5s28.5,12.3,28.5,31.2
			C331.4,618,318.8,630.6,302.3,630.6z M302.6,575.4c-12.6,0-20.7,9.9-20.7,24c0,14.4,8.4,24.3,21,24.3s20.7-9.9,20.7-24
			C323.6,585,315.2,575.4,302.6,575.4z"
          />
          <path
            fill="#FFFFFF"
            d="M351.8,629.1h-8.1v-59.5l8.1-1.8v12.6c3.9-7.5,9.6-12.6,17.4-12.6c3.9,0,7.5,1.2,10.5,2.7l-2.1,7.5
			c-2.7-1.5-5.7-2.7-9-2.7c-6.3,0-12,4.8-16.8,15.3V629.1L351.8,629.1z"
          />
          <path
            fill="#FFFFFF"
            d="M411.3,630.6c-18,0-29.1-12.9-29.1-31.2c0-18.6,12.3-31.5,29.1-31.5c7.2,0,13.5,1.8,18.6,5.1l-2.1,7.2
			c-4.5-3-10.2-4.8-16.5-4.8c-12.9,0-20.7,9.6-20.7,23.7c0,14.4,8.4,24,21,24c6,0,12-1.8,16.5-4.8l1.8,7.5
			C424.5,628.8,418.2,630.6,411.3,630.6z"
          />
          <path
            fill="#FFFFFF"
            d="M479.5,629.1v-38.5c0-10.5-4.2-15-12.6-15c-6.9,0-13.5,3.6-18.3,8.4v45.1h-8.1v-87.4l8.1-1.8
			c0,0,0,37.3,0,37.6c6.3-6.3,14.1-9.3,20.7-9.3c11.4,0,18.6,7.5,18.6,20.4v40.6H479.5z"
          />
        </g>
      </g>
    </svg>
  </h2>
  <p>
    <a href="https://pytorch.org/" target="_blank" rel="noreferrer">PyTorch</a>
    is Meta's flagship deep learning framework, which was originally inspired by
    Torch, a Lua based deep learning library. In the years after the initial release
    in 2016, PyTorch has become a favorite among deep learning researchers and has
    overtaken TensorFlow. PyTorch is often described as being more intuitive to Python
    developers than TensorFlow, you can say that it is more
    <em>pythonic</em> in some regards.
  </p>
  <div class="separator" />

  <h2>JAX</h2>
  <p>
    <a href="https://github.com/google/jax" target="_blank" rel="noreferrer"
      >JAX</a
    > is the new kid on the block. Like Keras and TensorFlow, JAX is developed by
    Google. The first release appeard in the year 2020 on github. JAX has been gaining
    a lot of traction over the last couple of years and there are even rumors that
    over time TensorFlow is going to be replaced by JAX.
  </p>
  <div class="separator" />

  <Alert type="warning">
    It is up to you which framework you choose. Especially when it comes to
    studying deep learning, all of the above frameworks are fine choices and are
    highly recommended. Keras and PyTorch have a much larger community and
    provide many more books and tutorials, but this might change within a couple
    of years. At the moment of writing most of our code examples are implemented
    in PyTorch.
  </Alert>
  <div class="separator" />
</Container>

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
    <a href="https://colab.research.google.com" target="_blank" rel="noreferrer"
      >Google Colab</a
    >
    (Colaboratory) provides you with free cloud compute in the form of Jupyter Notebooks.
    You can switch between CPU, GPU and TPU resources. If you use the platform extensively
    without a break they might switch you to a lower tier server or give you a time
    out, but considering the cost is zero, this is a great deal. Google also provides
    payed tiers with better GPUs and lower chance of getting disconnected. For learning
    purposes the free tier should be sufficient.
  </p>

  <p>
    <a href="https://www.kaggle.com" target="_blank" rel="noreferrer">
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

    is a plattform designed to run machine learning competitions. For that
    purpose they provide you weekly resources with at least 30 GPU hours and at
    least 30 TPU hours. Unlike Google Colab, Kaggle will not disconnect you, but
    when the hours run out you have to wait until the resources are reset. Kolab
    was acquired by Google, therefore the type of GPU or TPU you receive are
    comparable to those at Colab.
  </p>

  <p>
    If you are looking for alternatives free of Google,
    <a href="https://www.paperspace.com" target="_blank" rel="noreferrer"
      >Paperspace</a
    >
    might be it. Similar to Google Colab, Paperspace provides free cloud GPU notebooks
    for beginners and payed tiers for advanced users.
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
    <a href="https://lambdalabs.com" target="_blank" rel="noreferrer">
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
    <a href="https://www.linode.com" target="_blank" rel="noreferrer">
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
<h1>Education in Deep Learning</h1>
<div class="separator" />
<Container>
  <SvgContainer maxWidth={"600px"}>
    <Blackboard />
  </SvgContainer>
  <p>
    Before we take a deep dive into the foundations of deep learning we would
    like to talk about additional learning resources and to encourage you to
    consult many different sources on your deep learning journey. It usually
    requires more than one single resource to understand a topic. You should
    explore different books, videos, blog posts and research papers to get the
    full picture. Once you grasp the basics of deep learning, you still need to
    find a way to keep up to date with the newest research and this might not be
    the easiest task. Below we attempt to list the most relevant literature for
    beginners and give hints on ways you can stay informed regarding the newest
    developments in deep learning.
  </p>
  <div class="separator" />

  <h2>Books</h2>
  <p>
    Books are a great gateway into the world of deep learning. Especially when
    it comes to beginner level books the choices are great.
  </p>

  <div class="book-container">
    <img
      src="/books/geron_book.jpeg"
      alt="Book Cover Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow"
    />
    <p>
      "Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow" by
      Aurélien Géron is a great choice, especially if you don't know where to
      start. If you have only one book to pick, this is probabliy the right
      pick. At the moment of writing the second edition is available in online
      stores, but the third edition is currently being written and is due
      somewhere at the end of 3rd or the beginning the 4th quarter of 2022. The
      book covers classical machine learning and goes on to discuss deep
      learning with Keras and TensorFlow in detail. While the book is very well
      suited for beginners, advanced topics like handling large amounts of data
      are covered as well.
    </p>
  </div>
  <div class="book-container">
    <img
      src="/books/cholet_book.png"
      alt="Book Cover DEEP LEARNING with Python"
    />
    <p>
      The book "DEEP LEARNING with Python" is written by the creator of Keras,
      so you know you get quality content. François Chollet covers exclusively
      deep learning with a high focus on intuition and Python practice.
    </p>
  </div>

  <div class="book-container">
    <img
      src="/books/raschka_book.jpg"
      alt="Book Cover Machine Learning with PyTorch and Scikit-Learn"
    />
    <p>
      The book "Machine Learning with PyTorch and Scikit-Learn" by Sebastian
      Raschka is similar in scope to the book by Aurélien Géron, but instead of
      using Keras and TensorFlow the author uses PyTorch. For the time being
      World4AI is going to provide notebooks written in PyTorch, so you will
      probably feel most comfortable with this book if you plan on using our
      notebooks.
    </p>
  </div>
  <div class="separator" />

  <h2>Online Courses</h2>
  <p>
    If you learn better through visual and audio explanations, online courses
    might be a great fit. Just as with our book recommendations, there is a lot
    of quality content for beginners and few resources for intermediary and
    advanced users.
  </p>
  <p>
    <a href="https://fast.ai" target="_blank" rel="noreferrer">fast.ai</a> is several
    things at the same time. It is a non-profit with the goal of democratizing deep
    learning, a deep learning library based on PyTorch, a book called 'Deep Learning
    for Coders With Fasai and Pytorch' and an online course that teaches you deep
    learning. The approach taken by fast.ai differs from most books and courses.
    They teach you how you can generate state of the art results first and go into
    the basic building blocks of neural networks second.
  </p>
  <p>
    The deep learning <a
      href="https://www.deeplearning.ai/program/deep-learning-specialization/"
      target="_blank"
      rel="noreferrer">coursera</a
    > course by Andrew Ng is probably the most famous and the most acknowledged course
    in deep learning. The course starts with the basic building blocks of deep learning
    and steadily moves towards the state of the art. The course is not free, but
    there is financial support available if you can not afford the lectures.
  </p>

  <p>
    The <a
      href="https://cds.nyu.edu/deep-learning/"
      target="_blank"
      rel="noreferrer">NYU deep learning course</a
    > by Yann LeCun and Alfredo Canziani has become one of the most acknowledged
    courses over the last couple of years. The course is very in-depth and very engaging,
    but is probably not the best choice for absolute beginners. We recommend you
    take this course after you have had some exposure to an introductory course or
    book.
  </p>
  <div class="separator" />

  <h2>Deep Learning Practice</h2>
  <p>
    You can not learn deep learning without applying the algorithms with a deep
    learning framework yourself. Learning the theory is not going to be
    sufficient. If you have the data and an idea, then go for it and upload your
    project to GitHub. A popular alternative is to compete in machine learning
    challenges. The most popular plattform that host competitions is <a
      href="https://www.kaggle.com"
      target="_blank"
      rel="noreferrer">Kaggle</a
    >. There you can take part in discussions, upload datasets, create notebooks
    and take part in competitions. The plattform has a ranking system in the
    four mentioned categories and allows you to grow in the ranks and gain
    reputation and titles. Additionally many of the challenges give out monetary
    prices. Kaggle is essentially a social network, so if you want to get
    addicted to one, kaggle is a good choice for machine learning practitioners.
  </p>
  <div class="separator" />

  <h2>Keeping Up With Research</h2>
  <p>
    When you come from the outside of the deep learning research community, you
    do not have access to a personal network that you can utilize to find the
    most relevant research papers you should focus on. Fortunately you have
    access to twitter and can therefore follow companies that focus on deep
    learning research, established researchers and book authors to get access to
    your own individual network. Many of those sources will share interesting
    articles, tutorials, videos and the newest research. It will take some time
    to find which sources are the most reliable, but the benefits are enormous.
    We suggest that you create an account especially for the purpose of
    participating in the deep learning community and avoid other types of
    content. Otherwise you will experience the "dark side" of twitter.
  </p>
  <div class="separator" />
</Container>

<style>
  h2 {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  svg {
    margin: 0 10px;
  }

  .book-container {
    display: flex;
    margin-bottom: 20px;
  }
  .book-container img {
    width: 200px;
    margin-right: 30px;
    margin-bottom: 10px;
    flex-grow: 1;
    height: fit-content;
  }

  .book-container p {
    margin: 0;
  }

  @media (max-width: 700px) {
    .book-container {
      flex-direction: column;
    }
  }
</style>
