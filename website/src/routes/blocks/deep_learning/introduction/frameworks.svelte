<script>
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Frameworks</title>
  <meta
    name="description"
    content="Deep learning frameworks are necessary for an efficient workflow. At the moment TensorFlow, PyTorch and JAX dominate the market."
  />
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
    <a href="https://www.tensorflow.org/" target="_blank">TensorFlow</a> is
    Google's flagship deep learning framework. The framework was first released
    in the year 2015, which makes it the oldest in the list.
    <a href="https://keras.io/" target="_blank">Keras</a> is a higher level api that
    sits on top of TensorFlow and provides a lot of convenience for developers by
    reducing a lot of overhead. Keras was actually developed independently by François
    Chollet in order to facilitate fast and easy deep learning development. Originally
    Keras supported multiple deep learning libraries (for example Theano), but became
    an integral part of TensorFlow 2 in 2019 and nowadays only supports TensorFlow.
    You can say that TensorFlow and Keras go hand in hand. Keras is an incredible
    great choice for beginners. It has a vibrant community and you will have no difficulty
    to find tutorials and books to learn from.
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
    <a href="https://pytorch.org/" target="_blank">PyTorch</a> is Meta's
    flagship deep learning framework, which was originally inspired by Torch, a
    Lua based deep learning library. In the years after the initial release in
    2016, PyTorch has become a favorite among deep learning researchers and has
    overtaken TensorFlow. PyTorch is often described as being more intuitive to
    Python developers than TensorFlow, you can say that it is more
    <em>pythonic</em> in some regards.
  </p>
  <div class="separator" />

  <h2>JAX</h2>
  <p>
    <a href="https://github.com/google/jax" target="_blank">JAX</a> is the new kid
    on the block. Like Keras and TensorFlow, JAX is developed by Google. The first
    release appeard in the year 2020 on github. JAX has been gaining a lot of traction
    over the last couple of years and there are even rumors that over time TensorFlow
    is going to be replaced by JAX.
  </p>
  <div class="separator" />

  <p class="warning">
    It is up to you which framework you choose. Especially when it comes to
    studying deep learning, all of the above frameworks are fine choices and are
    highly recommended. Keras and PyTorch have a much larger community and
    provide many more books and tutorials, but this might change within a couple
    of years. At the moment of writing most of our code examples are implemented
    in PyTorch.
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
</style>
