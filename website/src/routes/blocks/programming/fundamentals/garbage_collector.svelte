<script>
  import Code from '$lib/Code.svelte';
  import GarbageCollection from '$lib/programming/fundamentals/variables/GarbageCollection.svelte';
  const helloWorldWithVariables2 = '>>> sentence = "Hello World!" \n>>> print(sentence) \nHello World!\n>>> sentence = "Goodbye World!"\n>>> print(sentence)\nGoodbye World!'
</script>
<h2>Garbage Collection</h2>
<Code code={helloWorldWithVariables2} />
<p>The above code snippet is the same that we used to explain that variables can be reassigned a different piece of data. When we reassign a variable it is possible that no variable points at a memory location. The object with the "Hello World!" text inside has no reference that points at it, yet it occupies some space in the memory of a computer. In other programming languages this "lost" data might constitute a problem, but Python has a so called garbage collector. Once a piece of data with no variable pointing at it is detected, it is automatically removed to clean up the memory. </p>
<div class="flex-center">
    <GarbageCollection />
</div>
<p class="info">A garbage collector cleans up the data that has no variable referencing it.</p>
<div class="separator"></div>
