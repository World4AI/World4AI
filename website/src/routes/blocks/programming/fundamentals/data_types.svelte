<script>
  import { fade } from 'svelte/transition';
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';   
  import Repl from '$lib/Repl.svelte';
</script>

<svelte:head>
    <title>World4AI | Programming | Data Types in Python</title>
    <meta name="description" content="Python differentiates between mutable and immutable data types. Numbers, strings and tuples are examples of immutable data types. List and dicionaries are mutable.">
</svelte:head>

<h1>Datatypes in Python</h1>
<Question>What data types are available in Python?</Question>
<div class="separator"></div>
<div class="flex-center">
<svg width="500" height="200" version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
 <g in:fade="{{duration: 1000}}" fill="var(--text-color)" fill-opacity=".99846" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width=".52917">
  <g stroke-width="1">
   <path d="m138.95 63.25 92.334-7.5499v64.3l-92.334-9.1514z"/>
   <path d="m90.71 58.409v58.308l48.245-5.8683v-47.599z"/>
   <path d="m90.71 116.72 100.29 14.91 40.289-11.627-92.334-9.1514z"/>
   <path d="m90.71 58.409 100.29-12.301 40.289 9.5921-92.334 7.5499z"/>
   <path d="m191 46.108v85.519l40.289-11.627v-64.3z"/>
   <path d="m15.06 62.638 100.29-12.301v85.519l-100.29-14.91z"/>
   <path d="m392.67 63.25 92.334-7.5499v64.3l-92.334-9.1514z"/>
   <path d="m344.42 58.409v58.308l48.245-5.8683v-47.599z"/>
   <path d="m344.42 116.72 100.29 14.91 40.289-11.627-92.334-9.1514z"/>
   <path d="m344.42 58.409 100.29-12.301 40.289 9.5921-92.334 7.5499z"/>
   <path d="m444.71 46.108v85.519l40.289-11.627v-64.3z"/>
   <path d="m268.77 62.638 100.29-12.301v85.519l-100.29-14.91z"/>
  </g>
  <circle cx="164.4" cy="89.704" r="15" fill="var(--background-color)" fill-opacity=".99846" stroke="black"/>
  <rect x="405" y="72" width="30" height="30" fill="var(--background-color)" fill-opacity=".99846" stroke="black"/>
 </g>
</svg>
</div>
<h2>What is a data type?</h2>
<p>In Python variables are used as a convenient label to reference the data, the variables are not the data itself. That data itself is represented by an object, which does not only hold the value of the data, but the type of the data. The data can be of different type, like a number or a string of text. Depending on the data type different attributes and methods are available to work with and process the data. For example numbers can be added together, while strings can be concatenated.</p>
<p class="info">The data type determines what properties the data posesses.</p>
<p>In Python data types are usually divided into <strong>mutable</strong> and <strong>unmutable</strong> types. Mutable means that the data that the variable references can be changed. Unmutable means the exact opposite, the data can not be changed, a new PyObject has to be created if we want to change the data.</p>
<p>This section  will focus on data types that are used most often in AI and machine learning, the below list of data types is not exhaustive, there are more types in Python. Additionally in this section we are only going to discuss what data types exist and some basic functionality. An in depth introduction will follow with individual sections that are specifically dedicated to some data types.</p>
<div class="separator"></div>

<h2>Immutable Data Types</h2>
<h3>String</h3>
<p>The string data type is designed to hold text data. A string can be created by surrounding a piece of text either by single quotes ' ' or by double quotes " ".</p>

<Code code={`string_1 = "Hello World!"
string_2 = 'Hello World!'
print(string_1)
print(string_2)
print(type(string_1))
print(type(string_2))`}/>
<Repl code={`Hello World!
Hello World!
<class 'str'>
<class 'str'>
`} />
<p>In the above code snippet we generate and print two strings, first using double quotes and then single quotes. To test out what type of data the variable references we use the <code>type()</code> function. This functions can be applied to all variables to test what type of an object the variable is referencing.</p>

<Code code={`print(len('Hello'))
string = 'Hello'
print(string[0])
print(string[4])
print(string[5])
`}/>

<Repl code={`5
'H'
'o'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: string index out of range
`} />
<p>The <code>len()</code> function can be used with data types that have some notion of length. For example <code>len('Hello')</code> will produce the number 5, because 'Hello' has exactly 5 letters. It also possible to access individual letters of a string by using square brackets [ ]. For example 'Hello'[0] will return the letter 'H'. In Python we use the number 0 to access the first letter, the number 1 to access the secod letter and so on. When we try to access a letter that is outside of the range of the string, it produces an error.</p>
<Code code={`string = 'Hello'
string[0] = 'B'
`} />
<Repl code={`Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment`} />
<p>It is easy to show that strings are immutable. When we try to change the letter at the 0's position, Python throws an error, indicating that assignment is not allowed for an immutable data type like a string.</p>

<h3>Numeric: Float and Integer</h3>
<p>There are several numeric data types in Python. Most notably and most relevant to machine learning are integers and floating point numbers. Integers are numbers without any fractions. For example 2, 3, 4. Floats on the other hand have some fractions associated with them, like 2.2 and 3.14. Even if you write 2.0, Python interprets the number as a float.</p>
<Code code={
`num1 = 2
num2 = 2.1
num3 = 2.0
print(type(num1))
print(type(num2))
print(type(num3))
`} />
<Repl code={`<class 'int'>
<class 'float'>
<class 'float'>
`} />
<p>For Python integers and floats are two distinct data types, as seen in the output of the <code>type()</code> function.</p>
<Code code={`num1 = 2
num2 = 2.2
print(num1 + num2)
print(num1 - num2)
print(num1 / num2)
print(num1 * num2)
`} />
<Repl code={`4.2
-0.20000000000000018
0.9090909090909091
4.4`} />
<p>You can apply the same basic mathematical operations to numerical values in Python as you would in a math exercise. You can add, subract, divide and multiply. You might ask yourself: "why did the subraction not return a value of exactly -0.2"? The answer has to do with the way how floating point numbers are represented in computer memory. This is considered a relatively advanced topic, for now remember that floating point numbers can not always be represented in the most exact way.</p>
<Code code={ `num1 = 1000
print(id(num1))
num1 = 1000 + 1
print(id(num1))
`} />
<Repl code={`139732329995248
139732329994928` } />
<p>There is often a confusion regarding the classification of numerical values as immutable data types. How can an integer be immutable if you can seemingly change the value of the integer? The reason for that classification can be found in the data snippet above. It is <strong>not</strong> possible to change the value of an integer in Python. Whenever you change the value by interacting with the variable that creates a new object with a new corresponding id. The <code>num1</code> variable is just the label that is connected to an object that can never be changed, but has to be recreated.</p>

<h3>Boolean</h3>
<p>A boolean can only take two possible values: <em>True</em> or <em>False</em>. Booleans are often used to control the flow of the program, for example to jump to a different point in code if the value of a variable is false.</p>
<Code code={`bool1 = True
bool2 = False
print(bool1)
print(bool2)
print(id(bool1))
print(id(True))
print(id(bool2))
print(id(False))
`} />
<Repl code={`True
False
94837591305216
94837591305216
94837591305184
94837591305184
`} />
<p>There is no reason to recreate True or False values each time you want to change the value. If there is only two option, both options can be kept in memory and the variables can just be pointed to the memory location. This behaviour can be observed in the code snippet above.</p>

<h3>Tuple</h3>
<p>A tuple is an immutable collection of objects. A tuple can be defined by surrounding values or objects by round brackets and separating them by commas. For example (1, 2, 3, 4) is a valid tuple. A different tuple can also be contained within a tuple, for example ((1, 2), (3, 4)).</p>
<Code code={
`tup1 = (1, 2, 3)
tup2 = (1, 'Name', True)
tup3 = (10, (10, 20))
print(tup1[0])
print(tup3[1][1])
tup1[0] = 4`} />
<Repl code={`1
20
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment`} />
<p>A tuple is in many ways similar to a string. You can access individual elements of the tuple by using square brackets, but changing the values inside the tuple is not allowed.</p>
<div class="separator"></div>

<h2>Mutable Data Types</h2>
<h3>List</h3>
<p>A list is a mutable collection of data. A list is generated by enclosing values in square brackets and separating those by commas. For example [1, 2, 3, 4] would generate a list in Python.</p>
<Code code={
`list1 = [1, 2, 3, 4]
print(list1)
print(list1[0])
list1[0] = 123
print(list1)
`} />
<Repl code={`[1, 2, 3, 4]
1
[123, 2, 3, 4]
`} />
<p>The code snippet above demonstrates that data within a list is mutable.</p>

<h3>Dictionary</h3>
<p>A dictionary is a mutable collection of key-value pairs. The name dictionary comes from the way the data structure works. Let's assume we have an english-spanish dictionary and we would like to have a translation for an english word. We use the english word as the key and receive the spanish translation, the value.</p>
<p> A dictionary is created by assigning each key a value. The key can later be used to get access to the value. The key-value pairs are enclosed in curly braces. For example {"{'key1':'value1', 'key2':2}"} creates a dictionary with two elements.</p> 
<Code code={
`dict1 = {'key1' : 'data1', 'key2' : 1}
print(dict1['key1'])
dict1['key1'] = 22
print(dict1)
`} />
<Repl code={`'data1'
{'key1': 22, 'key2': 1}
`} />
<p>Similar to other sequences in Python, elements can be accesses using square brackets. The difference between the list/tuple and the dictionary is the use the predefined keys instead of numbers.</p>
<div class="separator"></div>
