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
<div class="separator"></div>
<h2>Methods</h2>
<Code code={`text = ''
dir(text)`} />
<h3>Capitalize</h3>
<h3>Count</h3>
<h3>Find</h3>
<h3>Index</h3>
<h3>Join</h3>
<h3>Lower</h3>
<h3>Partition</h3>
<h3>Replace</h3>
<h3>Strip</h3>
<h3>Split</h3>
<h3>Title</h3>
<h3>Upper</h3>
<div class="separator"></div>
