<script>
  import {draw} from 'svelte/transition';
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
</script>

<svelte:head>
    <title>World4AI | Programming | Hello World</title>
    <meta name="description" content="To create a 'Hello World' program in Python we need to use the pring() function.">
</svelte:head>

<h1>Hello World</h1>
<Question>How do you output text to the console?</Question>
<div class="separator"></div>

<svg version="1.1" viewBox="0 0 270 140" xmlns="http://www.w3.org/2000/svg">
<g stroke="var(--text-color)" fill="none">
 <g id="letter-h"> 
  <path in:draw="{{duration: 500, delay: 0}}" d="m10 10v120"/>
  <path in:draw="{{duration: 500, delay: 0}}" d="m10 70h50"/>
  <path in:draw="{{duration: 500, delay: 0}}" d="m60 10v120"/>
 </g>
 <g id="leter-e">
  <path in:draw="{{duration: 500, delay: 0}}" d="m70 10v120"/>
  <path in:draw="{{duration: 500, delay: 0}}" d="m70 10h40"/>
  <path in:draw="{{duration: 500, delay: 0}}" d="m70 70h40"/>
  <path in:draw="{{duration: 500, delay: 0}}" d="m70 130h40"/>
 </g>
 <g id="letter-l-1">
  <path in:draw="{{duration: 500, delay: 0}}" d="m120 10v120h40" />
 </g>
 <g id="letter-l-2">
  <path in:draw="{{duration: 500, delay: 0}}" d="m170 10v120h40" /> 
 </g>
 <g id="letter-o">
  <path in:draw="{{duration: 500, delay: 0}}" d="m220 10v120h40v-120z" /> 
 </g>
</g>
</svg>
<h2>The very first program</h2>
<p>In every programming language there is a notion of a so called <em>Hello World</em> program. The only task of the program is to output some text, usually "Hello World!".  The program is used to get the first feel for the language, to test if the setup is working and to learn how to output simple text to the terminal.</p>
<p>In Python the code to output a simle "Hello World" text uses <code>print()</code>. The text we would like to output is set between the round brackets and the quotation marks indicate that we are working with a string of text.</p>
<Code code={'print("Hello World!")'} />
<Repl code={'Hello World!'} />
<p>As expected the code producces a 'Hello World!' text. Go ahead and copy the code, modify it, play with it. This is a great way to test if your setup is working correctly.</p>
<div class="separator"></div>
