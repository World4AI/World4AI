<script>
  import Question from '$lib/Question.svelte';
  import Repl from '$lib/Repl.svelte';
  import Operator from '$lib/Operator.svelte';
</script>

<h1>Boolean</h1>
<Question>What operators are useful for boolean values?</Question>
<div class="separator"></div>
<p>Operators that are most useful when it comes to booleans are so called logical operators. Their primarily use is to control the flow of the program by applying logical operations known from mathematics.</p>
<div class="separator"></div>

<h2>The and operator</h2>
<p>The <Operator>and</Operator> operator returns True when both operands are True and False otherwise. It is also possible to stack several "and" operations.</p>
<Repl code={`>>> a = True
>>> b = True
>>> c = False
>>> a and b
True
>>> a and c
False
>>> a and b and True
True
>>>
` } />
<div class="separator"></div>

<h2>The or operator</h2>
<p>The <Operator>or</Operator> operator returns True when one of the operands is True and False if both of the operands are False.</p>
<Repl code={`>>> a = True
>>> b = False
>>> c = False
>>> a or b
True
>>> b or c
False
`} />
<div class="separator"></div>

<h2>The not operator</h2>
<p>The <Operator>not</Operator> operator returns True when the operand is False and False when the operand is True.</p>
<Repl code={`>>> a = True
>>> not a
False
>>> b = False
>>> not b
True
`} />
<div class="separator"></div>
