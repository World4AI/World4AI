<script>
    import Code from '$lib/Code.svelte';
    import Question from '$lib/Question.svelte';
    const implicitNumbersCode = `
>>> a = 1
>>> b = 2.5
>>> type(a)
<class 'int'>
>>> type(b)
<class 'float'>
>>> c = a + b
>>> c
3.5
>>> type(c)
<class 'float'>
   >>>`
   const explicitNumbersCode = `
>>> a = 3.5
>>> b = int(a)
>>> b
3
>>> type(b)
<class 'int'>
>>>`
   const explicitStringCode = `
>>> a = '10'
>>> b = 'sentence'
>>> c = int(a)
>>> d = int(b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: invalid literal for int() with base 10: 'sentence'
>>>
`
   const conversionCode = `
>>> a = 10.5
>>> id(a)
139885775929744
>>> b = int(a)
>>> id(a)
139885775929744
>>> id(b)
139885777369680
>>>
`
</script>

<svelte:head>
    <title>World4AI | Programming | Casting</title>
    <meta name="description" content="In Python explicit or implicit casting can be used to recreate data with a different data type.">
</svelte:head>

<h1>Casting</h1>
<Question>What is casting and why is it useful?</Question>
<div class="separator"></div>
<h2>Definition</h2>
<p>Casting is the process of recreating data with a different data type. Sometimes casting is done implicitly, which means that Python does the casting without the explicit instruction of the programmer. Explicit casting on the other hand requires the programmer to indiciate which data and into what type has to be casted. Often the value remains the same, but sometimes there is some loss of information and the casted value is different from the original. </p>
<div class="separator"></div>
<h2>Implicit Casting</h2> 
<p>Implicit casting can only be done with data types that are compatible. For example an integer and a float are technically two different data types, but because the numeric operations between the two data types are defined, casting can be done.</p>
<Code code={implicitNumbersCode} />
<p>The code above creates two variables. Variable a points to an integer, while variable b points to a floating point number. The a + b operation triggers an implicit casting of a. The result is a once again a float. But why was this not done the other way around?  The result of c is 3.5, which we can expect from a summation of a floating point number and an integer. If b was casted to an integer the result would be a 2, because all the data after the dot would be cut off. Therefore implicit casting prefers the option without any loss of data.</p>
<div class="separator"></div>
<h2>Explicit Casting</h2>
<p>Explicit casting requires one of the predefined Python functions.</p>
<ul>
    <li><code>int()</code> casts into an integer.</li>
    <li><code>float()</code> casts into a floater.</li>
    <li><code>string()</code> casts into a string.</li>
    <li><code>bool()</code> casts into a boolean.</li>
</ul>
<Code code={explicitNumbersCode} />
<p>In the above example a floating point number is cast to an integer. This results in data loss.</p>
<Code code={explicitStringCode} />
<p>There is no guarantee, that explicit conversion is going to succeed. The code above shows that conversion is only possible, when it is logical. A string can for example be converted into a number if and only if the string contains the actual text representaion of an integer.</p>
<div class="separator"></div>
<h2>Casting is not Conversion</h2>
<p>Most definitions of casting state the following: "Casting is the converion of the variable's data type into another data type".</p>
<p>The definition is misleading. Below is a definition, that fits better the behaviour that Python exhibits.</p>
<p class="info"> Casting recreates the value of a variable or literal with a different data type.</p>
<Code code={conversionCode} />
<p>The code above shows that casting produces a new object with a different id, while the original object stay unchanged. The conversion is actually a recreation.</p>
<div class="separator"></div>
