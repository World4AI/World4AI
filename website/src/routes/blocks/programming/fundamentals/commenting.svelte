<script>
 import Question from '$lib/Question.svelte';
 import Code from '$lib/Code.svelte';
 import Repl from '$lib/Repl.svelte';
 import {draw} from 'svelte/transition';
 
 const singleLineComment = `# This is a single line comment
print("Hello World") 
 `
 const multiLineComment = '"""\nThis\nis\na\nmultiline\ncomment\n"""'
</script>

<svelte:head>
  <title>World4AI | Programming | Commenting</title>
  <meta name="description" content="">
</svelte:head>

<h1>Comments</h1>
<Question>Why are comments useful in Python?</Question>
<div class="separator"></div>
<div class="flex-center">
  <svg version="1.1" viewBox="0 0 200 200" width=200  height=200 xmlns="http://www.w3.org/2000/svg">
   <g fill="none" stroke="var(--text-color)" stroke-width="2">
    <path d="m40 20c-16.62 0-30 13.38-30 30v50c0 16.62 13.38 30 30 30h10v40l40-40h70c16.62 0 30-13.38 30-30v-50c0-16.62-13.38-30-30-30h-120z" stroke-linecap="round" stroke-linejoin="round"/>
    <g>
     <path in:draw="{{duration:1000}}" d="m30 35h130"/>
     <path in:draw="{{duration:1000, delay: 500}}" d="m30 50h145"/>
     <path in:draw="{{duration:1000, delay: 600}}" d="m30 65h105"/>
     <path in:draw="{{duration:1000, delay: 700}}" d="m30 80h130"/>
     <path in:draw="{{duration:1000, delay: 800}}" d="m30 95h90"/>
     <path in:draw="{{duration:1000, delay: 900}}" d="m30 110h115"/>
    </g>
   </g>
  </svg>
</div>
<p>Being able to comment your code is a basic skill that you need to posess before you learn the syntax of Python. A comment is a piece of code that is not being processed by Python. It's only function is the documentation of the written code. It is a way of communicating the code either to your collegues or yourself in the future.</p>
<p>If we want to comment only a single line of text, we use the # symbol (called hash or pound) in front of the comment. </p> 
<Code code={singleLineComment} />
<p>The line below the comment is not going to be affected, because # can only be used for single line comments.</p>
<Repl code={'Hello World'} />
<p>Oftentimes you will need to comment out whole blocks of code or to write down a more involved description. For that purpose you can use multiline comments. These are started and finished with three quotation marks <code>"""</code>. Anything between the quotation marks is disregarded by Python.</p>
<Code code={multiLineComment} />
<div class="separator"></div>
