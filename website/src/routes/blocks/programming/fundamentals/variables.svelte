<script>
  import {fade} from 'svelte/transition';
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  import PyObject from '$lib/PyObject.svelte';

</script>

<svelte:head>
    <title>World4AI | Programming | Variables </title>
    <meta name="description" content="In Python variables are used to access and modify data. Variables are names that allow you to work with objects in memory.">
</svelte:head>

<h1>Variables</h1>
<Question>What are variables and how are they used in Python?</Question>
<div class="separator"></div>
<div class="flex-center">
<svg width="500" height="200" version="1.1" viewBox="0 0 132.29 52.917" xmlns="http://www.w3.org/2000/svg">
 <g in:fade="{{duration: 1000}}">
  <g fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width=".67573">
   <g fill="var(--text-color)">
    <path d="m58.906 13.189 45.302-5.5562 18.198 4.3328-41.708 3.4103z" points="104.20716,7.6331293 122.40563,11.96593 80.697915,15.376241 58.905632,13.189379 " stroke="#000"/>
    <path d="m104.21 7.6331v38.629l18.198-5.2519v-29.044z" points="104.20716,46.262295 122.40563,41.010416 122.40563,11.96593 104.20716,7.6331293 "/>
    <path d="m58.906 13.189 45.302-5.5562v38.629l-45.302-6.7348z" points="104.20716,7.6331293 104.20716,46.262295 58.905632,39.527447 58.905632,13.189379 "/>
   </g>
  </g>
  <rect x="6.6146" y="18.521" width="25.135" height="13.229" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width=".52917"/>
  <path d="m33.073 25.135h23.812" fill="none" marker-end="url(#Arrow1Mend)" stroke="#fff" stroke-width=".21167"/>
 </g>
 <g fill="none" stroke="#000" stroke-width=".26458px">
  <path d="m9.2604 21.167v6.6146"/>
  <path d="m11.906 21.167v7.9375"/>
  <path d="m14.552 21.167v6.6146"/>
  <path d="m17.198 21.167v7.9375"/>
  <path d="m19.844 19.844v9.2604"/>
  <path d="m22.49 21.167v7.9375"/>
  <path d="m25.135 21.167v7.9375"/>
  <path d="m27.781 21.167v6.6146"/>
 </g>
</svg>
</div>

<h2>What are variables?</h2>
<p>In a programming language like Python we often deal with some data that has to be loaded, manipulated and generally accessed multiple times.</p>
<Code code={'print("Hello World!")'} />
<Repl code={'Hello World!'} />
<p>In the above example the string "Hello World!" is the data we are working with. Python creates a PyObject that holds "Hello World!", prints the text and discards the object right after. Yet this is not how programming usually looks like. In most cases you want to have access to the "Hello World!" string. You want to be able to modify it and to possibly access the modified value over and over again. For that purpose variables can be used. A variable allows a programmer to use a descriptive name to access and possibly modify the data.</p> 
<p class="info">Variables are names that allow programs to access and modify the data in the computer memory.</p>
<Code code={`number = 42
print(number)
`} />                                                                                                          
<Repl code={'42'} />
<p>In the example above we create a variable named "number" with a value of 42. In the next step the variable is used to print the value in the terminal.</p>
<PyObject variable={'number'} />
<p>The variable "number" is used to access the PyObject with the value of 42. Therefore when we talk about variables, what we actually talk about is a convenient name, that helps the programmer to find the corresponding data. The data is located somewhere in your computer memory and keeping track of it without a human readable tag would be impracticle. Internally the address is usually reprsented by a hexadecimal number like 0x7fff1275c05c, which is wildly inconvenient. The variables in Python are like pointers to the location of the data and not the data itself.</p>
<Code code={`number = 42 
number_2 = number
print(number_2)
print(id(number))
print(id(number_2))`}/>
<Repl code={`
42
139657570418736
139657570418736
`} />
<p>The above example shows how variables work in Python. In the first line an object is created and the variable <code>number</code> points at the address of the memory location of the number 42. In the second line an additional variable is created that points to the same memory location, therefore no additonal PyObject needs to be created. When we use the <code>print()</code> function the  same content is output in the terminal. To show that this is actually the same object we can use the <code>id()</code> function. When we apply that function to a variable this function returns the unique integer id of that object. For number and number_2 the id is exactly the same, therefore we are dealing with the same object.</p>
<PyObject variable={'number'} variable2={'number_2'} />
<p>The diagram exemplifies that behaviour. The nuber 42 is stored somewhere in the computer memory and both variables point at the same memory object.</p>
<p class="info">In Python a variable is a convenient name for a memory location of the data.</p>
<Code code={`sentence = "Hello World!"
print(sentence) 
sentence = "Goodbye World!"
print(sentence)
`} />
<Repl code={`Hello World!
Goodbye World!`} />
<p>The term <strong>variable</strong> starts to make a lot of sense once you realize how variables actually work. Variables are named for their ability to be assigned different values (PyObjects). Simply put, variables are <em>variable</em> (meaning flexible) in what they can contain. In the above example the <code>sentence</code> variable is first assigned the value "Hello World!" and then reassigned a value "Goodbye World!".</p>
<div class="separator"></div>
<h2>Variable Names</h2>
<p>There are certain rules you should  follow when you name your variables. On the one side there are some strict rules that you absolutely have obey or your Python programm will not run. On the other side some variable naming convetions are considered more Pythonic than others, you should follow them unless you have a good reason not to.</p>
<p>Variable names can contain any letters [a-z] or [A-Z], any numbers [0-9] and an underline _, but variable names can not start with a number.</p> 
<Code code={`valid_variable_name = 10
print(valid_variable_name`} />
<Repl code={'10'} />
<p>The above code works as expected. We use a valid variable name and print the value to the terminal.</p>
<Code code={'1_variable = 2'} />
<Repl code={'SyntaxError: invalid decimal literal'} />
<p>A variable name that starts with a number is not valid Python syntax, therefore we get a syntax error.</p>
<p>Additionally there are some reserved words in Python that can not be used as variable names. These words are core elements of the language and you will encounter them as we progress though this block of lectures. If you are curious you can input <code>help('keywords')</code> into the prompt or your text editor to display the reserved words.</p>
<Repl code={`>>> help('keywords')`} />
<Repl code={`Here is a list of the Python keywords.  Enter any keyword to get more help.

False               break               for                 not
None                class               from                or
True                continue            global              pass
__peg_parser__      def                 if                  raise
and                 del                 import              return
as                  elif                in                  try
assert              else                is                  while
async               except              lambda              with
await               finally             nonlocal            yield`} />
<p>If you try and use one of the above keywords as a varaible name you will get a syntax error.</p>
<p>There are several syntactically correct ways of naming your variables when the name consists of multiple words.</p>
<ul> 
  <li>Camel Case starts with a lower case letter and each word afterwards starts with an upper case letter. For example: thisIsCamelCase  </li>
  <li>Snake Case consists only of lower letter words and each word is separated by an underscore. For esample: this_is_snake_case.</li>
</ul>
<p class="info">If possible use snake case for variables names.</p>
<div class="separator"></div>
