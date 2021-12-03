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
<svg width="500" height="200" version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <marker id="Arrow1Mend" overflow="visible" orient="auto">
   <path transform="rotate(180) translate(10)" d="m0 0 5-5-17.5 5 17.5 5-5-5z" fill="var(--text-color)" fill-rule="evenodd" stroke="context-stroke" stroke-width="1pt"/>
  </marker>
 </defs>
 <g in:fade="{{duration: 1000}}" fill="#fff" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.3475">
  <g fill="var(--text-color)">
   <path d="m229.65 40.771 169.35-20.771 68.031 16.197-155.92 12.749z" points="399,20.000001 467.03109,36.197251 311.11568,48.945973 229.64988,40.770855 " stroke="#000"/>
   <path d="m399 20v144.41l68.031-19.633v-108.58z" points="399,164.40691 467.03109,144.77388 467.03109,36.197251 399,20.000001 "/>
   <path d="m229.65 40.771 169.35-20.771v144.41l-169.35-25.177z" points="399,20.000001 399,164.40691 229.64988,139.23011 229.64988,40.770855 "/>
  </g>
 </g>
 <g in:fade="{{duration: 1000}}" transform="translate(0,5)" stroke="#000">
  <rect x="15" y="70" width="95" height="50" fill="var(--text-color)" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/>
  <g transform="matrix(3.7795 0 0 3.7795 -10 -1.1375e-6)" fill="none" stroke-width=".26458px">
   <path d="m9.2604 21.167v6.6146"/>
   <path d="m11.906 21.167v7.9375"/>
   <path d="m14.552 21.167v6.6146"/>
   <path d="m17.198 21.167v7.9375"/>
   <path d="m19.844 19.844v9.2604"/>
   <path d="m22.49 21.167v7.9375"/>
   <path d="m25.135 21.167v7.9375"/>
   <path d="m27.781 21.167v6.6146"/>
  </g>
 </g>
 <path d="m120 100h100" fill="none" marker-end="url(#Arrow1Mend)" stroke="#fff" stroke-width="1px"/>
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
