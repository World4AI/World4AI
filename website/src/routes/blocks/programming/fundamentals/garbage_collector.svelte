<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  import PyObject from '$lib/PyObject.svelte';
</script>

<h1>Garbage Collection</h1>
<Question>What is garbage collection?</Question>
<div class="separator"></div>

<p>A garbage collector is a mechanism many modern programming languages have implemented in order to clean up untracked objects.</p>
<Code code={`number = 420 
print(number) 
number = 2400
print(number)
`} />
<Repl code={`420
2400`} />
<p>In the above code snippet we assign a variable with the name "number" a value of 42. Under the hood a PyObject is generated and the label "number" is the way we can interract with that object. Each PyObject requires a certain amount of memory in order to be stored.  So when we reassign a value of 2400 to "number" no variable references the PyObject with the value 420. In other programming languages this "lost" object might constitute a problem, but Python has a so called garbage collector. Once a PyObject without any variables is detected, it is automatically removed to clean up the memory. </p>
<p class="info">A garbage collector cleans up the data that has no variable referencing it.</p>
<PyObject variable={'number'} value={420} value2={2400} garbageCollection={true}/>
<p>Internally each PyObject has a count, which allows the garbage collector to mark PyObjects for removal. Once a count has reached a value of 0, that means there is no variable associated with the object and the garbage collector can clean up the memory.</p>
<div class="separator"></div>
