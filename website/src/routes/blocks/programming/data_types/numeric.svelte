<script>
    import Question from '$lib/Question.svelte';
    import Code from '$lib/Code.svelte';
    import Math from '$lib/Math.svelte';
    const plusOperatorCode = `
>>> a = 1 + 1
>>> a
2
>>>`
    const minusOperatorCode = `
>>> a = 10 - 5
>>> a
5
>>>
`
    const multiplyOperatorCode = `
>>> a = 2.3 * 5
>>> a
11.5
>>>
`
    const divideOperatorCode = `
>>> a = 5 / 2
>>> a
2.5
>>>
`
    const floorOperatorCode = `
>>> a = 5 // 2
>>> a
2
>>>
`
    const moduloOperatorCode = `
>>> a = 5 % 2
>>> a
1
>>>
`
    const exponentOperatorCode = `
>>> a = 2 ** 3
>>> a
8
>>>
`
    const simpleAssignmentCode = `
>>> a = 10
>>> b = a
>>> b
10
>>>
`
    const reassignmentCode = `
>>> a = 1
>>> a
1
>>> a = 2
>>> a
2
>>>
`
   const assignmentShorthandCode = `
>>> a = 1
>>> a += 1
>>> a
2
>>> b = 1
>>> b -= 1
>>> b
0
>>> c = 1
>>> c *= 2
>>> c
2
>>> d = 5
>>> d /= 2
>>> d
2.5
>>> e = 5
>>> e //= 2
>>> e
2
>>> f = 5
>>> f %= 2
>>> f
1
>>> g = 2
>>> g **= 3
>>> g
8
>>>
`
</script>

<h1>Numeric Data Types</h1>
<Question>What arithmetic and assignment operations are possible with numeric data types?</Question>
<div class="separator"></div>

<p>Any operations that are even remotely related to mathematics require the use of numeric data types. In machine learning each picture, sound, text and so on has to be transfered into a numeric data type, before we can apply any algorithms to the data. </p>
<div class="separator"></div>
<h2>Arithmetic Operators</h2>
<p>Arithmetic operators are designed to handle the most atomic mathematical operations you could imagine, mostly we are talking about elementary school math. The more complex operations are often just compositions of these operations, even neural networks are for the most part just additions and multiplications. This is how much of programming in Python works. Complex operations are build upon more and more basic operations, until the most atomic level is reached. We need therefore to master simple arithmetic, before we move on.</p>
<p>The plus operator <span class="operator">+</span> sums the two operands and returns the result.</p>
<Code code={plusOperatorCode} />
<p>The minus operator <span class="operator">-</span> subtracts the second operand from the first operand and returns the result.</p>
<Code code={minusOperatorCode} />
<p>The multiplication operator <span class="operator">*</span> multiplies the two operands and returns the result.</p>
<Code code={multiplyOperatorCode} />
<p>The divison operator <span class="operator">/</span> divides the first operand by the second operand and returns the result.</p>
<Code code={divideOperatorCode} />
<p>The floor divison operator <span class="operator">//</span> divides the first operand by the second operand and cuts off the remainder of the dvision.</p>
<Code code={floorOperatorCode} />
<p>The modulo operator <span class="operator">%</span> is the counterpart to the floor division. It throws away the quotient and returns the remainder.</p>
<Code code={moduloOperatorCode} />
<p>The exponentiation operator <span class="operator">**</span> takes the left operand to the power of the right operand.</p>
<Code code={exponentOperatorCode} />
<div class="separator"></div>
<h2>Assignment Operators</h2>
<p>Assignment operators assign values to the left operand. The right side of the operator either contains the address to the already existing object in memory or the value with the new memory address needs to be generated.</p>
<Code code={simpleAssignmentCode} />
<p>In the code snippet above the value is generated in the first line and assigned to a. In the second line no new value is generated. Both variables show at the same address.</p>
<p>The assignment operator in Python <span class="operator">=</span> works differently than the mathematical equal sign known from math. In mathematics writing <Math latex={'a = 1'} /> and <Math latex={'a = 2'} /> is not allowed, because the equal sign implies equality and not assignment. Essentially that equality would mean <Math latex={'1 = 2'} />, which is nonsense. </p>
<Code code={reassignmentCode} />
<p>In Python the same operation would imply reassignment, which is valid.</p>
<p>All other numeric assignment operators in Python are just shorthand notations. They combine arithmetic and assignment operators into one single operator. The <span class="operator">+=</span> for example is the same operation as taking the value that is associated with the left operand adding it to the value of the right operand and reassigning the sum to the variable on the left sign. For example <span>a+=5</span> is the same operation as <span>a = a + 5. </span></p>
<p>The same logic applies to the following operators: <span class="operator">-=</span>, <span class="operator">*=</span>, <span class="operator">/=</span>, <span class="operator">//=</span>, <span class="operator">%=</span>, <span class="operator">**=</span>.</p>
<Code code={assignmentShorthandCode} />
<div class="separator"></div>
<style>
    .operator {
        line-height: 1rem;
        display: inline-block;
        margin: 0;
        padding: 9px 10px;
        border-bottom: 1px solid var(--main-color-2);
    }
</style>

