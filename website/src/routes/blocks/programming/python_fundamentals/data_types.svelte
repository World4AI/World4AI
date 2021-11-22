<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';   
  const stringCode = `
>>> string_1 = "Hello World!"
>>> string_2 = 'Hello World!'
>>> print(string_1)
Hello World!
>>> print(string_2)
Hello World!
>>> type(string_1)
<class 'str'>
>>> type(string_2)
<class 'str'>
`
 const stringCode2 = `
>>> len('Hello')
5
>>> string = 'Hello'
>>> string[0]
'H'
>>> string[4]
'o'
>>> string[5]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: string index out of range
>>>
`
const stringCode3 = `
>>> string = 'Hello'
>>> string[0] = 'B'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment
>>>
`
const numericCode = `
>>> num1 = 2
>>> num2 = 2.1
>>> num3 = 2.0
>>> type(num1)
<class 'int'>
>>> type(num2)
<class 'float'>
>>> type(num3)
<class 'float'>
>>>
`
const numericCode2 = `
>>> num1 = 2
>>> num2 = 2.2
>>> num1 + num2
4.2
>>> num1 - num2
-0.20000000000000018
>>> num1 / num2
0.9090909090909091
>>> num1 * num2
4.4
>>>
`
const numericCode3 = `
>>> num1 = 1000
>>> id(num1)
139732329995248
>>> num1 = 1000 + 1
>>> id(num1)
139732329994928
>>>
`
const boolCode = `
>>> bool1 = True
>>> bool2 = False
>>> bool1
True
>>> bool2
False
>>> id(bool1)
94837591305216
>>> id(True)
94837591305216
>>> id(bool2)
94837591305184
>>> id(False)
94837591305184
>>>
`
const tupleCode = `
>>> tup = (1, 2, 3)
>>> tup1 = (1, 2, 3)
>>> tup2 = (1, 'Name', True)
>>> tup1[0]
1
>>> tup1[0] = 4
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>>
`
const listCode = `
>>> list1 = [1, 2, 3, 4]
>>> list1
[1, 2, 3, 4]
>>> list1[0]
1
>>> list1[0] = 123
>>> list1
[123, 2, 3, 4]
>>>
`
const dictCode = `
>>> dict1 = {'key1' : 'data1', 'key2' : 1}
>>> dict1['key1']
'data1'
>>> dict1['key1'] = 22
>>> dict1
{'key1': 22, 'key2': 1}
>>>
`
</script>

<svelte:head>
    <title>World4AI | Programming | Data Types in Python</title>
    <meta name="description" content="Python differentiates between mutable and immutable data types. Numbers, strings and tuples are examples of immutable data types. List and dicionaries are mutable.">
</svelte:head>

<h1>Datatypes in Python</h1>
<Question>What data types are available in Python?</Question>
<div class="separator"></div>
<h2>What is a data type?</h2>
<p>In Python variables are used as a convenient label to reference the data, the variables are not the data itself. That data itself is represented by an object, which does not only hold the value of the data, but the type of the data. The data can be of different type, like a number or a string of text. Depending on the data type different options are available to work with and process the data. For example numbers can be added together, while strings can be concatenated.</p>
<p class="info">The data type determines what properties the data posesses.</p>
<p>In Python the data types are usually divided into mutable and unmutable types. Mutable means that the data that the variable references can be changed. Unmutable means the exact opposite, the data can not be changed.</p>
<p>The below list of data types is not exhaustive, there are more types in Python. This section  will focus on data types that occur most often in AI and machine learning. Additionally in this section we are only going to discuss what data types exist and some basic functionality. An in depth introduction will follow with individual sections that are specifically dedicated to some data types.</p>
<div class="separator"></div>
<h2>Immutable Data Types</h2>
<h3>String</h3>
<p>The string data type is designed to hold text data. A string can be created by surrounding a piece of text either by single quotes ' ' or by double quotes " ".</p>
<Code code={stringCode} />
<p>In the above code snippet we generate and print two strings, first using double quotes and then single quotes. To test out what type of data the variable references we use the <code>type()</code> function. This functions can be applied to all variables to test what type of an object the variable is referencing.</p>
<Code code={stringCode2} />
<p>The <code>len()</code> function can be used with data types that have some notion of length. For example <code>len('Hello')</code> will produce the number 5, because 'Hello' has exactly 5 letters. It also possible to access individual letters of a string by using square brackets [ ]. For example 'Hello'[0] will return the letter 'H'. In Python we use the number 0 to access the first letter, the number 1 to access the secod letter and so on. When we try to access a letter that is outside of the range of the string, it produces an error.</p>
<Code code={stringCode3} />
<p>It is easy to show that strings are immutable. When we try to access the letter at the position #0, the Python interpreter throws an error, indicating that assignment is not allowed for an immutable data type like string.</p>
<h3>Numeric: Float and Integer</h3>
<p>There are several numeric data types in Python. Most notably and most relevant to machine learning are integers and floating point numbers. Integers are numbers without any fractions. For example 2, 3, 4. Floats on the other hand have some fractions associated with them, like 2.2 and 3.14. Even if you write 2.0, Python interprets the number as a float.</p>
<Code code={numericCode} />
<p>For Python integers and floats are two distinct data types, as seen in the output of the <code>type()</code> function.</p>
<Code code={numericCode2} />
<p>You can apply the same basic mathematical operations to numerical values in Python as you would in a math exercise. You can add, subract, divide and multiply.</p>
<Code code={numericCode3} />
<p>There is often a confusion regarding the classification of numerical values as immutable data types. How can an integer be immutable if you can seemingly change the value of the integer? The reason for that classification can be found in the data snippet above. It is <strong>not</strong> possible to change the value of an integer in Python. Whenever you change the value by interacting with the variable that creates a new object with a new corresponding id. The <code>num1</code> variable is just the pointer to an object that can never be changed, but has to be recreated.</p>
<h3>Boolean</h3>
<p>A boolean can only take two possible values: <em>True</em> or <em>False</em>. Booleans are often used to control the flow of the program, for example to jump to a different point in code if the value of a variable is false.</p>
<Code code={boolCode} />
<p>There is no reason to recreate True or False values each time you want to change the value. If there is only two option, both options can be kept in memory and the variables can just be pointed to the memory location. This behaviour can be observed in the code snippet above.</p>
<h3>Tuple</h3>
<p>A tuple is an immutable collection of objects. A tuple can be defined by surrounding values by round brackets and separating them by commas. For example (1, 2, 3, 4) is a valid tuple.</p>
<Code code={tupleCode} />
<p>A tuple is in many ways similar to a string. You can access individual elements of the tuple by using square brackets, but changing the values inside the tuple is not allowed.</p>
<h2>Mutable Data Types</h2>
<h3>List</h3>
<p>A list is a mutable collection of data. A list is generated by enclosing values in square brackets and separating those by commas. For example [1, 2, 3, 4] would generate a list in Python.</p>
<Code code={listCode} />
<p>The code snippet above demonstrates that data within a list is mutable.</p>
<h3>Dictionary</h3>
<p>A dictionary is a mutable collection of key-value pairs. A dictionary is created by assigning each value a key, which can later be used to get access to the value. The key-value pairs are enclosed in curly braces. For example {"{'key1':'value1', 'key2':2}"} creates a dictionary with two elements.</p> 
<Code code={dictCode} />
<p>Similar to other sequences the elements can be accesses using square brackets. The difference between the list/tuple and the dictionary is the use the predefined keys instead of numbers.</p>
