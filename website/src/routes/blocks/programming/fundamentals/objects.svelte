<script>
  import Question from '$lib/Question.svelte';
  import PyObject from '$lib/PyObject.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
</script>

<h1>Objects</h1>
<Question>What are objects?</Question>
<div class="separator"></div>
<div class="flex-center">
<svg width="200" height="200" version="1.1" viewBox="0 0 52.917 52.917" xmlns="http://www.w3.org/2000/svg">
 <g stroke="var(--text-color)">
  <circle cx="26.458" cy="26.458" r="22.534" fill="var(--main-color-1)" fill-opacity=".99846" stroke-linecap="round" stroke-linejoin="round" stroke-width=".26458"/>
  <g fill="none" stroke-width=".26458px">
   <path d="m0 0h52.917"/>
   <path d="m0 0 52.917 1.3229"/>
   <path d="m0 0 52.917 2.6458"/>
   <path d="m0 0 52.917 3.9688"/>
   <path d="m0 0 52.917 5.2917"/>
   <path d="m0 0 52.917 6.6146"/>
   <path d="m0 0 52.917 7.9375"/>
   <path d="m0 0 52.917 9.2604"/>
   <path d="m0 0 52.917 10.583"/>
   <path d="m0 0 52.917 11.906"/>
   <path d="m0 0 52.917 13.229"/>
   <path d="m0 0 52.917 14.552"/>
   <path d="m0 0 52.917 15.875"/>
   <path d="m0 0 52.917 17.198"/>
   <path d="m0 0 52.917 18.521"/>
   <path d="m0 0 52.917 19.844"/>
   <path d="m0 0 52.917 21.167"/>
   <path d="m0 0 52.917 22.49"/>
   <path d="m0 0 52.917 23.812"/>
   <path d="m0 0 52.917 25.135"/>
   <path d="m0 0 52.917 26.458"/>
   <path d="m0 0 52.917 27.781"/>
   <path d="m0 0 52.917 29.104"/>
   <path d="m0 0 52.917 30.427"/>
   <path d="m0 0 52.917 31.75"/>
   <path d="m0 0 52.917 33.073"/>
   <path d="m0 0 52.917 34.396"/>
   <path d="m52.917 35.719"/>
   <path d="m0 0 52.917 35.719"/>
   <path d="m0 0 52.917 37.042"/>
   <path d="m0 0 52.917 38.365"/>
   <path d="m0 0 52.917 39.688"/>
   <path d="m0 0 52.917 41.01"/>
   <path d="m0 0 52.917 42.333"/>
   <path d="m0 0 52.917 43.656"/>
   <path d="m0 0 52.917 44.979"/>
   <path d="m0 0 52.917 46.302"/>
   <path d="m0 0 52.917 47.625"/>
   <path d="m0 0 52.917 48.948"/>
   <path d="m0 0 52.917 50.271"/>
   <path d="m0 0 52.917 51.594"/>
   <path d="m0 0 52.917 52.917"/>
   <path d="m0 0 51.594 52.917"/>
   <path d="m0 0 50.271 52.917"/>
   <path d="m0 0 48.948 52.917"/>
   <path d="m0 0 47.625 52.917"/>
   <path d="m0 0 46.302 52.917"/>
   <path d="m0 0 44.979 52.917"/>
   <path d="m0 0 43.656 52.917"/>
   <path d="m0 0 42.333 52.917"/>
   <path d="m0 0 41.01 52.917"/>
   <path d="m0 0 39.688 52.917"/>
   <path d="m0 0 38.365 52.917"/>
   <path d="m0 0 37.042 52.917"/>
   <path d="m0 0 35.719 52.917"/>
   <path d="m0 0 34.396 52.917"/>
   <path d="m0 0 33.073 52.917"/>
   <path d="m0 0 31.75 52.917"/>
   <path d="m0 0 30.427 52.917"/>
   <path d="m0 0 29.104 52.917"/>
   <path d="m0 0 27.781 52.917"/>
   <path d="m0 0 26.458 52.917"/>
   <path d="m0 0 25.135 52.917"/>
   <path d="m0 0 23.812 52.917"/>
   <path d="m0 0 22.49 52.917"/>
   <path d="m0 0 21.167 52.917"/>
   <path d="m0 0 19.844 52.917"/>
   <path d="m0 0 18.521 52.917"/>
   <path d="m0 0 17.198 52.917"/>
   <path d="m0 0 15.875 52.917"/>
   <path d="m0 0 14.552 52.917"/>
   <path d="m0 0 13.229 52.917"/>
   <path d="m0 0 11.906 52.917"/>
   <path d="m0 0 10.583 52.917"/>
   <path d="m0 0 9.2604 52.917"/>
   <path d="m0 0 7.9375 52.917"/>
   <path d="m0 0 6.6146 52.917"/>
   <path d="m0 0 5.2917 52.917"/>
   <path d="m0 0 3.9688 52.917"/>
   <path d="m0 0 2.6458 52.917"/>
   <path d="m0 0 1.3229 52.917"/>
   <path d="m0 0v52.917"/>
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
