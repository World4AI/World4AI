<script>
  import {fade} from 'svelte/transition';
  import { onMount } from 'svelte';
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  import PyObject from '$lib/PyObject.svelte';
  let direction = 'east';

  let offsetTop = 0;
  let offsetBottom = 0;

  onMount(() => {
       let interval = setInterval(() => {
          if(direction == 'east') {
            direction = 'north';
          } else if (direction == 'north'){
            direction = 'west';
          } else if (direction == 'west'){
            direction = 'south';
          } else if (direction == 'south'){
            direction = 'east'; 
          }
        }, 1900);

       return () => clearInterval(interval);
  })
</script>

<svelte:head>
    <title>World4AI | Programming | Variables </title>
    <meta name="description" content="In Python variables are used to access and modify data. Variables are names that allow you to work with objects in memory.">
</svelte:head>

<h1>Variables</h1>
<Question>What are variables and how are they used in Python?</Question>
<div class="separator"></div>
<div class="flex-center">
<svg width="500" height="500" version="1.1" viewBox="0 0 132.29 132.29" xmlns="http://www.w3.org/2000/svg">
 <g id="variable" stroke-width=".26458">
  <circle cx="66.146" cy="66.146" r="17.905" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="50.915756" y="68.460426" fill="var(--text-color)" font-family="sans-serif" font-size="6.35px" style="line-height:1.25" xml:space="preserve"><tspan x="50.915756" y="68.460426" fill="#000000" font-size="6.35px" stroke-width=".26458">VARIABLE</tspan></text>
 </g>
 <g fill="var(--text-color)" stroke="var(--text-color)" stroke-width=".52917">
  {#if direction == 'east'}
    <path in:fade="{{duration: 1000}}" id="east-direction" d="m87.313 66.146h26.458"/>
  {:else if direction == 'south'}
    <path in:fade="{{duration: 1000}}" id="south-direction" d="m66.146 87.313v26.458"/>
  {:else if direction == 'west'}
    <path in:fade="{{duration: 1000}}" id="west-direction" d="m44.979 66.146h-26.458"/>
  {:else if direction =='north'}
    <path in:fade="{{duration: 1000}}" id="north-direction" d="m66.146 44.979v-26.458"/>
  {/if}
 </g>
 {#if direction == 'east'}
 <g in:fade="{{duration: 1000}}" id="east-variable" stroke-width=".26458">
  <rect x="116.42" y="59.531" width="13.229" height="13.229" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="119.57153" y="70.003464" font-family="sans-serif" font-size="10.583px" style="line-height:1.25" xml:space="preserve"><tspan x="119.57153" y="70.003464" stroke-width=".26458">1</tspan></text>
 </g>
 {:else if direction == 'north'}
 <g in:fade="{{duration: 1000}}" id="north-variable" stroke-width=".26458">
  <rect x="59.531" y="2.6458" width="13.229" height="13.229" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="62.921234" y="13.187814" font-family="sans-serif" font-size="10.583px" style="line-height:1.25" xml:space="preserve"><tspan x="62.921234" y="13.187814" stroke-width=".26458">2</tspan></text>
 </g>
 {:else if direction == 'west'}
 <g in:fade="{{duration: 1000}}" id="west-variable" stroke-width=".26458">
  <rect x="2.6458" y="59.531" width="13.229" height="13.229" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="6.0358181" y="70.073227" font-family="sans-serif" font-size="10.583px" style="line-height:1.25" xml:space="preserve"><tspan x="6.0358181" y="70.073227" stroke-width=".26458">3</tspan></text>
 </g>
 {:else if direction == 'south'}
 <g in:fade="{{duration: 1000}}" id="south-variable" stroke-width=".26458">
  <rect x="59.531" y="116.42" width="13.229" height="13.229" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="62.799797" y="126.88372" font-family="sans-serif" font-size="10.583px" style="line-height:1.25" xml:space="preserve"><tspan x="62.799797" y="126.88372" stroke-width=".26458">4</tspan></text>
 </g>
 {/if}
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
