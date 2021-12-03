<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  import { fade } from 'svelte/transition';
</script>

<svelte:head>
    <title>World4AI | Programming | Operators</title>
    <meta name="description" content="Python has built in operators. Operators act like functions, which take operands as input and create ouput.">
</svelte:head>

<h1> Operators in Python </h1>
<Question>What is an operator?</Question>
<div class="separator"></div>
<div in:fade="{{duration: 1000}}" class="flex-center">
  <svg width="500" height="200" version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
   <g transform="matrix(.76754 0 0 .76608 -78.922 21.569)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width="3.0614">
    <g fill="var(--text-color)">
     <path d="M 115.65,40.771 285,20 353.031,36.197 197.111,48.946 Z" points="285,20.000001 353.03109,36.197251 197.11568,48.945973 115.64988,40.770855 " stroke="#000" stroke-width="5.2064"/>
     <path d="m285 20v144.41l68.031-19.633v-108.58z" points="285,164.40691 353.03109,144.77388 353.03109,36.197251 285,20.000001 " stroke-width="3.0614"/>
     <path d="M 115.65,40.771 285,20 V 164.41 L 115.65,139.233 Z" points="285,20.000001 285,164.40691 115.64988,139.23011 115.64988,40.770855 " stroke-width="3.0614"/>
    </g>
   </g>
   <g transform="matrix(.76754 0 0 .76608 217.81 21.569)" fill="#fff" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width="3.0614">
    <g fill="var(--text-color)">
     <path d="M 115.65,40.771 285,20 353.031,36.197 197.111,48.946 Z" points="285,20.000001 353.03109,36.197251 197.11568,48.945973 115.64988,40.770855 " stroke="#000" stroke-width="5.2064"/>
     <path d="m285 20v144.41l68.031-19.633v-108.58z" points="285,164.40691 353.03109,144.77388 353.03109,36.197251 285,20.000001 " stroke-width="3.0614"/>
     <path d="M 115.65,40.771 285,20 V 164.41 L 115.65,139.233 Z" points="285,20.000001 285,164.40691 115.64988,139.23011 115.64988,40.770855 " stroke-width="3.0614"/>
    </g>
   </g>
   <g fill="none" stroke-width="2" stroke="var(--text-color)">
    <path d="m250 70v40" />
    <path d="m230 90h40" />
    <path d="m225 65h50v50h-50z" />
   </g>
 </svg>
</div>

<h2>Definition</h2>
<p>Operators are symbols or keywords that are predefined by Python. They act like atomic functions that take some input and generate some output. In fact true Python functions are often just compositions of these atomic functions, where you stack operation after operation.</p>
<Code code={`result = 10 + 5
print(result)`} />
<Repl code={`15`} />
<p>In the example above the + operator takes the two numbers as input, adds them together and returns the sum. The numbers or variables to the left and to the right of the operator are called operands. In the example above the number 10 and the number 5 are operands.</p>
<Code code={`a = 5`} />
<p>The assignment operator = is responsible for assigning values to variables.</p>
<p>There are many more predefined Python operators, but covering them in detail without context does not make a lot of sense. Some operators like + work on many different data types like numbers, lists and strings, while the modulo operator % is only useful when working with numbers. Therefore operators will be introduced one after another in sections dedicated to specific data types.</p>

<h2>Operator Types</h2>
<p>Most Python operators are symbols. We can get a list of all available symbols by using the help() function.</p>
<Code code={`>>> help('symbols')`} />
<Repl code={`Here is a list of the punctuation symbols which Python assigns special meaning
to. Enter any symbol to get more help.

!=                  +                   <=                  __
"                   +=                  <>                  \`
"""                 ,                   ==                  b"
%                   -                   >                   b'
%=                  -=                  >=                  f"
&                   .                   >>                  f'
&=                  ...                 >>=                 j
'                   /                   @                   r"
'''                 //                  J                   r'
(                   //=                 [                   u"
)                   /=                  \\                   u'
*                   :                   ]                   |
**                  <                   ^                   |=
**=                 <<                  ^=                  ~
*=                  <<=                 _
`} />

<p>Not all symbols are operators and not all symbols can work on their own as operators. For example brackets like ( ) or [ ] only work in pairs.</p>
<p>Other operators are keywords. For example: <code>and, or, not, is, is not, in, not in.</code></p>
<div class="separator"></div>
