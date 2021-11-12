<script>
    import Code from '$lib/Code.svelte';
    const operationExample = `
>>> 10 + 5
15
>>>`;
    const assignmentExample = `
>>> a = 5
>>>`;
</script>
<h1> Operators in Python </h1>
<p><em>What is an operator?</em></p>
<div class="separator"></div>
<h2>Definition</h2>
<p>Operators are tokens / symbols that are predefined by Python. They act like atomic functions that take some input and generate some output. In fact true Python functions are essentialy just compositions of these atomic functions, where you stack operation after operation.</p>
<Code code={operationExample} />
<p>In the example above the + operator takes the two numbers as input, adds them together and returns the sum (or the pointer that shows at the memory address). </p>
<Code code={assignmentExample} />
<p>The = operator is responsible for assigning values to variables. Here we could imagine that this token takes the number 5 as input, generates the value in the memory and eventually returns a memory address of the value to the variable.</p>
<p>There are many more predefined Python operators, but inroducing them without context does not make a lot of sense. Some operators like + work on many different data types like numbers, lists and strings, while the modulo operator % is only useful when working with numbers. Therefore operators will be introduced one after another in sections dedicated to specific topics.</p>
<p>Below is a selected list of python operators. Most of them are represented by symbols, some are words.</p>
<p>+, -, *, /, %, **, //, =, +=, -=, *=, /=, %=, **=, ==, !=, &gt;, &lt;, &gt;=, &lt;=, and, or, not, [ ], is, is not, in, not in</p>
<div class="separator"></div>
