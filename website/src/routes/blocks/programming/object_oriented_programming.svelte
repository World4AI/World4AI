<script>
  import Question from "$lib/Question.svelte";
  import Repl from "$lib/Repl.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Object-Oriented Programming</title>
  <meta
    name="description"
    content="Python is an object-oriented programming language. Even simple data types like integers and strings are objects."
  />
</svelte:head>

<h1>Object Oriented Programming (OOP)</h1>
<Question>Is Python an object oriented language?</Question>
<div class="separator" />

<p>
  Everything in Python is an object. Even basic built-in data types like
  integers and strings are objects of the class <Highlight>str</Highlight> and <Highlight
    >int</Highlight
  > respectively.
</p>
<Repl
  code={`
>>> my_text = 'Hello World'
>>> type(my_text)
<class 'str'>
>>> my_number = 42
>>> type(my_number)
<class 'int'>
  `}
/>
<p>
  The power of Python and other object oriented languages comes from the ability
  to create user defined classes and to build custom objects from those. What
  the sentence exactly means and how we can put the sentence into practice with
  Python is going to be the topic of the next chapters.
</p>
<div class="separator" />
