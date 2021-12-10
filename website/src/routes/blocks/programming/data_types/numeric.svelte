<script>
  import Question from "$lib/Question.svelte";
  import Code from "$lib/Code.svelte";
  import Repl from "$lib/Repl.svelte";
  import Math from "$lib/Math.svelte";
  import Operator from "$lib/Operator.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Numeric Datatypes</title>
  <meta
    name="description"
    content="Python has built-in numeric data types like integers, floats and complex numbers. Many mathematical operators, assignment operators and comparison operators can be used with numeric data types."
  />
</svelte:head>

<h1>Numeric Data Types</h1>
<Question>What operators are available for numeric data types?</Question>
<div class="separator" />

<p>
  Any operations that are even remotely related to mathematics require the use
  of numeric data types. In machine learning each picture, sound, text and so on
  has to be transfered into a numeric data type, before we can apply any
  algorithms to the data. Python provides different types of operators that can
  be utilized to conduct mathematical calculations.
</p>
<div class="separator" />
<h2>Operators</h2>
<h3>Arithmetic Operators</h3>
<p>
  Arithmetic operators are designed to handle the most atomic mathematical
  operations you could imagine, mostly we are talking about elementary school
  math.
</p>

<h4>Addition</h4>
<p>
  The plus operator <Operator><Math latex={"+"} /></Operator> sums the two operands
  and returns the result.
</p>
<Code
  code={`a = 1 + 1
print(a)`}
/>
<Repl code={"2"} />

<h4>Subtraction</h4>
<p>
  The minus operator <Operator><Math latex={"-"} /></Operator> subtracts the second
  operand from the first operand and returns the result.
</p>
<Code
  code={`a = 10 - 5
print(a)`}
/>
<Repl code={"5"} />

<h4>Multiplication</h4>
<p>
  The multiplication operator <Operator><Math latex={"*"} /></Operator> multiplies
  the two operands and returns the result.
</p>
<Code
  code={`a = 2.3 * 5
print(a)`}
/>
<Repl code={"11.5"} />

<h4>Division</h4>
<p>
  The divison operator <Operator><Math latex={"/"} /></Operator> divides the first
  operand by the second operand and returns the result.
</p>
<Code
  code={`a = 5 / 2
print(a)`}
/>
<Repl code={"2.5"} />

<h4>Floor Division</h4>
<p>
  The floor divison operator <Operator><Math latex={"//"} /></Operator> divides the
  first operand by the second operand and cuts off the remainder of the dvision.
</p>
<Code
  code={`a = 5 // 2
print(a)`}
/>
<Repl code={"2"} />

<h4>Modulo</h4>
<p>
  The modulo operator <Operator><Math latex={"\\%"} /></Operator> is the counterpart
  to the floor division. It throws away the quotient and returns the remainder.
</p>
<Code
  code={`a = 5 % 2
print(a)`}
/>
<Repl code={"1"} />

<h4>Exponentiation</h4>
<p>
  The exponentiation operator <Operator><Math latex={"**"} /></Operator> takes the
  left operand to the power of the right operand.
</p>
<Code
  code={`a = 2 ** 3
print(a)`}
/>
<Repl code={"8"} />
<div class="separator" />

<h3>Assignment Operators</h3>
<p>
  Assignment operators assign values of the right operand to the left operand.
</p>
<Code
  code={`a = 10
b = a
print(b)`}
/>
<Repl code={"10"} />

<p>
  In the code snippet above both variables are assigned the same object in
  memory using the assignement operator <Operator><Math latex={"="} /></Operator
  >.
</p>

<p>
  All other numeric assignment operators in Python are just shorthand notations.
  They combine arithmetic and assignment operators into one single operator. The <Operator
    >+=</Operator
  > operator for example combines addition and assignment. For example
  <span>a+=5</span>
  is the same operation as <span>a = a + 5. </span>
</p>
<p>
  The same logic applies to the following operators: <Operator>-=</Operator>, <Operator
    >*=</Operator
  >, <Operator>/=</Operator>, <Operator>//=</Operator>, <Operator>%=</Operator>, <Operator
    >**=</Operator
  >.
</p>
<Code
  code={`
a = 1
a += 1
print(a)

b = 1
b -= 1
print(b)

c = 1
c *= 2
print(c)

d = 5
d /= 2
print(d)

e = 5
e //= 2
print(e)

f = 5
f %= 2
print(f)

g = 2
g **= 3
print(g)`}
/>
<Repl
  code={`2
0
2
2.5
2
1
8
`}
/>
<div class="separator" />
<h3>Comparison Operators</h3>
<p>
  Comparison operators allow us to compare two numbers. This operators allow us
  to answer questions like: "Is number a larger than number b?" or "Is number a
  equal to number b?". The result is always a boolean, a true or false value.
</p>
<h4>Equality and Inequality</h4>
<p>
  Often we want to figure out if two numbers are equal or unequal. When we talk
  about equality we talk about equality of values and not equality of objects.
  As long as the objects contain the same values, they are considered to be
  equal. The equality operator <Operator><Math latex={"=="} /></Operator> returns
  True if both objects contain the same value. For example the expression 5==5 returns
  True and 5==4 returns False. The inequality operator <Operator
    ><Math latex={"!="} /></Operator
  > does the exact opposite. The expression 1!=1 returns False and 1!=2 returns True.
</p>
<Code
  code={`a = 5
b = 5
c = 4
print(a==b)
print(a==c)
`}
/>
<Repl
  code={`True
False`}
/>
<h4>Greater and Less</h4>
<p>
  When we want to figure out which of the two numbers is larger or smaller, we
  use either the greater than operator <Operator><Math latex={">"} /></Operator>
  or smaller than operator <Operator><Math latex={"<"} /></Operator>. For
  example 2 &lt; 1 would return False and 2 &gt; 1 would return True.
</p>
<p>
  Sometimes it is desirable to know if one of the numbers is larger or equal to
  the second number. In that case we use either <Operator
    ><Math latex={">="} /></Operator
  > or <Operator><Math latex={"<="} /></Operator>.
</p>
<Code
  code={`a = 2
b = 1
print(a > b)
print(a < b)
`}
/>
<Repl
  code={`True
False
`}
/>
<div class="separator" />
