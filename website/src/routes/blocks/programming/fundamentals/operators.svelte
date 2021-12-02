<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  const assignmentExample = `
>>> a = 5
>>>`;
</script>

<svelte:head>
    <title>World4AI | Programming | Operators</title>
    <meta name="description" content="Python has built in operators. Operators act like functions, which take operands as input and create ouput.">
</svelte:head>

<h1> Operators in Python </h1>
<Question>What is an operator?</Question>
<div class="separator"></div>

<h2>Definition</h2>
<p>Operators are tokens / symbols that are predefined by Python. They act like atomic functions that take some input and generate some output. In fact true Python functions are often just compositions of these atomic functions, where you stack operation after operation.</p>
<Code code={`result = 10 + 5
print(result)`} />
<Repl code={`15`} />
<p>In the example above the + operator takes the two numbers as input, adds them together and returns the sum (or the pointer that shows at the memory address). </p>
<Code code={`a = 5`} />
<p>The assignment operator = is responsible for assigning values to variables.</p>
<p>There are many more predefined Python operators, but covering them in detail without context does not make a lot of sense. Some operators like + work on many different data types like numbers, lists and strings, while the modulo operator % is only useful when working with numbers. Therefore operators will be introduced one after another in sections dedicated to specific data types.</p>
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
