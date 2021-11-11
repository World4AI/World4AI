<script>
    import Highlight from "svelte-highlight";
    import python from "svelte-highlight/src/languages/python";
    import "svelte-highlight/src/styles/github-dark.css";
  
    const dynamicCode = `
>>> number = 10
>>> type(number)
<class 'int'>
>>> number = 'Hello'
>>> type(number)
<class 'str'>
>>>
`  
    const strongCode = `
>>> number = 6
>>> number2 = 6.1
>>> number + number2
12.1
>>> string = 'Hello'
>>> number + string
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for +: 'int' and 'str'
>>>
`
</script>

<svelte:head>
    <title>World4AI | Programming | Dynamic and Strong Typing</title>
    <meta name="description" content="Python is a dynamically-typed and a strongly-typed language.">
</svelte:head>

<h1>Dynamic and Strong Typing</h1>
<p><em>What is dynamic typing and what is strong typing?</em></p>
<div class="separator"></div>
<h2>Dynamic Typing</h2>
<p>Python is classified as a dynamically-typed language. In a dynamically-typed language the programmer does not have to declare the type of a variable. The type is deduced during the run of the program. Additionally thy type that the variable points to can change during the execution of a program.</p>
<Highlight language="{python}" code="{dynamicCode}" />
<p>In the code snippet above the variable is created pointing at a number but ends up pointing at a string.</p>
<div class="separator"></div>
<h2>Strong Typing </h2>
<p>Python is classified as strongly-typed language. Essentially this means that operations accross incompatible types of data are not allowed to happen. For example you can add an integer to a floating point number, but adding a string to an integer is prohibited in Python.</p>
<Highlight language="{python}" code="{strongCode}" />
<p>If you try to add add a string to a number, Python will throw an error.</p>
<div class="separator"></div>
