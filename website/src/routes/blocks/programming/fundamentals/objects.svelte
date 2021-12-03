<script>
  import { fade } from 'svelte/transition';
  import Question from '$lib/Question.svelte';
  import PyObject from '$lib/PyObject.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
</script>

<h1>Objects</h1>
<Question>What are objects?</Question>
<div class="separator"></div>

<div class="flex-center">
 <svg width="500" height="200" version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
  <g in:fade="{{duration:1000}}" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.3475">
   <g fill="var(--text-color)">
    <path d="m115.65 40.771 169.35-20.771 68.031 16.197-155.92 12.749z" points="285,20.000001 353.03109,36.197251 197.11568,48.945973 115.64988,40.770855 " stroke="#000"/>
    <path d="m285 20v144.41l68.031-19.633v-108.58z" points="285,164.40691 353.03109,144.77388 353.03109,36.197251 285,20.000001 "/>
    <path d="m115.65 40.771 169.35-20.771v144.41l-169.35-25.177z" points="285,20.000001 285,164.40691 115.64988,139.23011 115.64988,40.770855 "/>
   </g>
  </g>
 </svg>
</div>
<p>It is not customary to start learning Python with the introduction of objects. But in Python this is almost unavoidable, because almost everything you encounter in Python is an object. For example each number and each piece of text are objects. And while we can teach Python without using the word object, we are still going to encounter objects from the very beginning. Therefore we might as well cover the very basics and the terminology and leave the more complex stuff for later, when the need arises.</p>
<p class="info">Everything in Python is an object!</p>
<p>So if everything is an object, what exactly is an object? We are going to look at the definition from two different perspectives. The first perspective is going to deal with the internal Python object representation, called PyObject. The second perspective is going to cover attributes and methods.</p>
<div class="separator"></div>
<h2>PyObject</h2>
<p>The Python interpreter can theoretically be implemented in different programming languages. The reference implementation is done in the programming language C. It is the most used implementation and what we are going to learn applies mostly to that particular implementation, called CPython.</p>
<p>Whenever we say that everything in Python is an object, what we actually mean is that a PyObject in C has to be created whenever we touch something in Python. When we work with a piece of text or a number, a PyObject (or a relative of PyObject) is what we actually deal with under the hood.</p>
<Code code={'print(42)'} />
<p>The above code generates a PyObject with the following characteristics.</p>
<PyObject />
<p>The object is of the type integer with the value of 42. The count of 1 shows that print() is the only "thing" that is accessing the PyObject.</p>
<p>The type, the value and the count and it's significance are going to be covered in future sections. For now you only have to remember that everyting in Python is an object that is represented by a PyObject under the hood.</p>
<div class="separator"></div>
<h2>Attributes and Methods</h2>
<p>The word object comes from object oriented programming. The intuition behind object oriented programming (oop) states that we can model real world objects with some data (attributes) and some functionality (methods). For example a particular duck would have some weight and color as attributes and would have swim() and quack() as methods.</p>
<p>In future chapters we will learn how to create our own types of objects, for now we will deal with simple built in objects like integers and strings of texts. Built in Python objects like strings of texts or numbers have attributes and methods as well, which depend on the type of object.</p>
<p>The method upper() for example turns the whole text into uppercase letters.</p>
<Code code={'"quack".upper()'} />
<Repl code={'QUACK'} />
<p class="info">Objects possess attributes and methods.</p>
<div class="separator"></div>
