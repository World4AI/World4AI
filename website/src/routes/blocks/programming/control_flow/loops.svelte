<script>
  import Question from "$lib/Question.svelte";
  import Repl from "$lib/Repl.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<h1>Loops</h1>
<Question
  >What are loops and what types of loops are available in Python?</Question
>
<div class="separator" />

<p>
  In the example below we need to execute the function <Highlight
    >say_hello()</Highlight
  > five times in a row. And the only way we know how to perform that task is to
  actually write the function call five times.
</p>
<Repl
  code={`
>>> def say_hello():
...     print('Hello')
... 
>>> say_hello()
Hello
>>> say_hello()
Hello
>>> say_hello()
Hello
>>> say_hello()
Hello
>>> say_hello()
Hello
>>>`}
/>
