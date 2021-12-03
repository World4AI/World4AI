<script>
  import { fade } from 'svelte/transition';
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  import PyObject from '$lib/PyObject.svelte';
</script>

<h1>Garbage Collection</h1>
<Question>What is garbage collection?</Question>
<div class="separator"></div>
<div class="flex-center">
  <svg width="500" height="200" version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
    <g in:fade="{{duration: 1000}}" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.3475">
    <g fill="var(--text-color)">
     <path d="m115.65 40.771 169.35-20.771 68.031 16.197-155.92 12.749z" points="285,20.000001 353.03109,36.197251 197.11568,48.945973 115.64988,40.770855 " stroke="#000"/>
     <path d="m285 20v144.41l68.031-19.633v-108.58z" points="285,164.40691 353.03109,144.77388 353.03109,36.197251 285,20.000001 "/>
     <path d="m115.65 40.771 169.35-20.771v144.41l-169.35-25.177z" points="285,20.000001 285,164.40691 115.64988,139.23011 115.64988,40.770855 "/>
    </g>
   </g>
   <g fill="none">
    <circle cx="202.89" cy="89.266" r="33.499" marker-end="url(#TriangleInM)"  stroke="var(--background-color)" stroke-dasharray="3, 9" stroke-linecap="square" stroke-width="3"/>
    <g stroke="var(--text-color)" stroke-width="2">
     <path d="m5 90h105" stroke-dasharray="2,2"/>
     <path d="m50 80 20 20"/>
     <path d="m70 80-20 20"/>
    </g>
   </g>
  </svg>
</div>
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
