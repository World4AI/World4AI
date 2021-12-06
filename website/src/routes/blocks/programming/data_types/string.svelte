<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  import Operator from '$lib/Operator.svelte';
  import Math from '$lib/Math.svelte';
</script>

<h1>String</h1>
<Question>How can strings be processed in Python?</Question>
<div class="separator"></div>
<p>Working with text is an essential skill in machine learning. Python offers different ways of proccessing strings. Some of the operators that work with numbers can also be applied to strings. But there are also a ton of built in string methods that can be extremely useful.</p>
<div class="separator"></div>

<h2>Operators</h2>
<h3>Concatenation (Addition)</h3>
<p>Similar to numbers we can use addition operators like <Operator><Math latex={'+'} /></Operator> or <Operator><Math latex={'+='} /></Operator> with strings. These operations concatenate strings. For example 'a'+'b'+'c' would generate 'abc'.</p>
<p></p>
<Code code={`text1 = 'yes'
text2 = 'or'
text3 = 'no'
result = text1 + ' ' + text2 + ' ' + text3
print(result)`}/>
<Repl code={"'yes or no'"} />

<h3>Repetition (Multiplication)</h3>
<p>It is also possible to repeat the same string several times using multiplication operators like <Operator><Math latex={'*'} /></Operator> or <Operator><Math latex={'*='} /></Operator>. For exmple 3 * 'a' would produce 'aaa'.</p>
<Code code={`text = '*' * 5
print(text)
text = 'a' * 3
print(text)
text = 'ab' * 2
print(text)
`} />
<Repl code={`'*****'
'aaa'
'abab'
`} />

<h3>Comparisons</h3>
<p>The same comparison operators that work with numbers also work with strings. But what does it mean to apply operators like <Operator><Math latex={'>'} /></Operator> <Operator><Math latex={'<'} /></Operator> <Operator><Math latex={'='} /></Operator> to strings?</p>
<p>When Python compares two characters like 'a' and 'b', Python actually compares two numbers, the unicode code points of those characters. To find out those code points we can utilize the <code>ord()</code> function.</p>
<Code code={`code_1 = ord('a')
code_2 = ord('b')
print(code_1)
print(code_2)
print('b' > 'a')
`} />
<Repl code={`97
98
True
`} />
<p>The code point for 'a' is 97 and the code point for 'b' is 98, therefore 'b' > 'a' results in True.</p>
<p>When we compare two strings that are more than 1 charachter long, the comparison is done 1 letter at a time. Therefore 'Abc' &lt; 'Zbc' is True.</p>
<Code code={`name_1 = 'Alexander'
name_2 = 'Zoe'
name_3 = 'Alexander'

print(name_1 == name_2)
print(name_1 == name_3)
print(name_1 < name_2)
`} />
<Repl code={`False
True
True
`} />

<h3>Membership</h3>
<p>Often it is important to figure out if a certain piece of string (e.g. a word) is contained in a sentence. For that we can use the <Operator>in</Operator> operator. The operator returns a boolean value, which results in True when the substring is contained within a string.</p>
<Code code={`text = 'What are you doing?'
print('What' in text)
print('How' in text)`} />
<Repl code={`True
False`} />

<h3>Slicing</h3>
<p>The slicing operator <Operator>[start:stop:step]</Operator> allows us to get a substring from a string using the address of the starting letter, the address of the ending letter and the the step size.</p>
<p>For example 'Hello World!'[0:5:1] returns 'Hello', where 0 is the starting address (letter 'H') and 5 is the ending address. The ending address is not inclusive, therefore the last letter that is used in the slicing operation is at the position 4, which is the letter 'o'. The stepping size is the increment that is used from the starting until the ending position. If the increment is 2 for example, then we skip every second position and would return letters at positions 0, 2 and 4.</p>
<p>When the starting position is 0, we can skip the number 0 and write 'Hello World!'[:5:1] instead. When the increment is 1 we can skip the increment notation alltogether and just write 'Hello World!'[:5]. Similarly 'Hello World'[6:] would return each character from position 6 until the last character.</p>
<Code code={`text = 'Hello Cruel World!'
print(text[0:5])
print(text[:5])
print(text[6:11]
print(text[6:])
`} />
<Repl code={`'Hello'
'Hello'
'Cruel'
'Cruel World!'
`} />
<p>When we work with sequences we are allowed to use a negative index, which counts from the end of the sequence. The index of -1 for example would return the last letter of the sentence.</p>
<Code code={`text = 'Hello Cruel World!'
print(text[-1])
print(text[-6:])
`} />
<Repl code={`'!'
'World!'
`} />
<p>The below example shows how the steping size influences the outcome of the slicing operation.</p>
<Code code={`text = 'Hello Cruel World!'
print(text[::2])
`} />
<Repl code={`'HloCulWrd'`} />
<div class="separator"></div>

<h2>Formatting</h2>
<p>Usually strings that need to be printed are not hardcoded, but come at least partially from outside variables. Those variables often need to be printed in the desired format using either the <code>format()</code> method or so called f-strings.</p>
<h3>Format Method</h3>
<p>The format method searches for placeholders marked by curly braces &#123; &#125; and replaces them with provided variables or values. For example '&#123; &#125; to apples'.format('yes') would result in 'yes to apples'.</p>
<p>Similarly as shown in the example below we can utilize variables to replace placeholders.</p>
<Code code={`text = 'I have {} apples'
number = 10
print(text)
formatted_text = text.format(number)
print(formatted_text)`}/>
<Repl code={`I have {} apples 
I have 10 apples`} />
<p>As shown in the code below we can provide more than one placeholder to be replaced by variables.</p>
<Code code={`
fruits = 'apples'
number = '10'
sentence = 'I have {} {}'
formatted_sentence = sentence.format(number, fruits)
print(formatted_sentence)
`} />
<Repl code={`I have 10 apples`} />

<p>The code snippet above utilizes so called positional arguments. That means that the first placeholder is replaced by the first variable and the second placeholder is replaced by the second variable. We can write the same code in a more explicit way by indicating which placeholder is going to be replaced by which variable. For example '&#123;0&#125; are &#123;1&#125;'.format('apples', 'awesome') would produce the string 'apples are awesome' while '&#123;1&#125; are &#123;0&#125;'.format('apples', 'awesome') would produce 'awesome are apples'.</p>
<Code code={`fruits = 'apples'
number = 10
sentence_1 = 'I have {0} {1}'
sentence_2 = 'I have {1} {0}'
formatted_sentence_1 = sentence_1.format(number, fruits)
formatted_sentence_2 = sentence_2.format(number, fruits)
print(formatted_sentence_1)
print(formatted_sentence_2)
`} />
<Repl code={`I have 10 apples
I have apples 10
`} />
<p>Additionally we can use named placeholders. For example '&#123;fruits&#125;'.format(fruits='apples') would produce the string 'apples'.</p>
<Code code={`sentence = 'I have {number} {fruits}'
formatted_sentence = sentence.format(number=10, fruits='apples')
print(formatted_sentence)
`} />
<Repl code={`I have 10 apples`} />
<p>There are many options when it comes to formatting and it is almost impossible to remember them all. We are going to cover some basic example to get the feel for what is possible.</p>
<p>The colon : indicates the start of formatting.</p>
<Code code={`>>> '{:}'.format(1000) 
'1000'`}/> 
<p>The number after the colon shows the length of the string.</p>
<Code code={`>>> '{:10}'.format(1000) 
'      1000'`}/> 
<p>The comma "," makes sure that the comma is used for a thousand separator.</p>
<Code code={`>>> '{:,}'.format(1000) 
'1,000'`}/> 
<p>The "f" indicates that we are dealing with a floating point numbers.</p>
<Code code={`>>> '{:f}'.format(1000) 
'1000.000000'`}/> 
<p>The number after the dot "." indicates the precision of a floating point number.</p>
<Code code={`>>> '{:.2f}'.format(1000) 
'1000.00'`}/> 
<h3>f-strings</h3>
<p>The f-strings are a newer method of dealing with string formatting. Essentially f-strings add some convenience when dealing with string literals. F-strings are initiated by the letter 'f' followed by a string. For example f'{10}' produces the string '10'.</p>
<Code code={`
number = 10
fruits = 'apples'
sentence = f'I have {number} {fruits}'
print(sentence)
`} />
<Repl code={'I have 10 apples'} />
<p>Essentially with f-strings the code gets cleaner, as we can avoid the trailing format() method.</p>
<div class="separator"></div>

<h2>Methods</h2>
<p>String methods provide additional useful functionalities. As strings are not mutable all those methods take the original text as input and return a new PyObject, while the original string ramains unchanged.</p>

<h3>Capitalize</h3>
<p>The capitalize() method returns a string, where the first letter of the input string is made uppercase.</p>
<Repl code={`>>> 'hello'.capitalize()
'Hello'
`} />

<h3>Count</h3>
<p>The count() method returns an integer, which indicates how many times a certain substring is found within the input string.</p>
<Code code={`>>> 'hello'.count('h')
1
>>> 'hello'.count('l')
2
>>> 'hello'.count('a')
0
>>> 'hello'.count('lo')
1
`} />

<h3>Find</h3>
<p>The find() method searches for a substring and returns the index of the first occurence of that substring. If the substring is not found, find() returns -1.</p>
<Code code={`>>> 'hello'.find('l')
2
>>> 'hello'.find('a')
-1
`} />

<h3>Join</h3>
<p>Join is a convenient method to concatenate strings within a sequence. The string whose join() method is called is added between the sequence of strings.</p>
<Code code={`>>> '-'.join('abc')
'a-b-c'
>>> '*'.join(['a', 'b', 'c'])
'a*b*c'
>>> '*'.join(['ab', 'bc', 'cd'])
'ab*bc*cd'
`} />

<h3>Lower</h3>
<p>The lower() method makes each single letter lowercase.</p>
<Code code={`>>> 'HELLO'.lower()
'hello'
`} />

<h3>Replace</h3>
<p>The replace(a, b) method looks for a substring a and replaces that substing with substing b.</p>
<Code code={`>>> 'hello'.replace('h', 'b')
'bello'
>>> 'hello'.replace('l', 's')
'hesso'
>>> 'hello'.replace('ll', 'sa')
'hesao'
`} />

<h3>Strip</h3>
<p>The strip() method removes whitespace to the left and the right of the string.</p>
<Code code={`>>> '   hello   '.strip()
'hello'
`} />

<h3>Split</h3>
<p>The split() method splits a string into individual components and puts them into a list. The "sep" argument is used to determine the substring that is used for separation. The standard argument is the empty string ' '.</p>
<Code code={`
>>> 'hello world'.split()
['hello', 'world']
>>> 'hello-world'.split(sep='-')
['hello', 'world']
`} />

<h3>Title</h3>
<p>The title() method makes the first letter of each individual word uppercase.</p>       
<Code code={`>>> 'hello world'.capitalize()
'Hello world'
>>> 'hello world'.title()
'Hello World'
`} />

<h3>Upper</h3>
<p>The upper() method makes each individual letter uppercase.</p>
<Code code={`>>> 'hello world'.upper()
'HELLO WORLD'
`} />
<div class="separator"></div>
       
