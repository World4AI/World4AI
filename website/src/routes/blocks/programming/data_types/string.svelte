<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
</script>

<h1>String</h1>
<Question>How can strings be processed in Python?</Question>
<div class="separator"></div>
<p>Python offeres different ways of proccessing strings. Some of the operators that work with numbers can also be applied to strings. But there are also a ton of built in string methods that can be extremely useful.</p>
<div class="separator"></div>

<h2>Operators</h2>
<h3>Concatenation (Addition)</h3>
<Code code={`text1 = 'yes'
text2 = 'or'
text3 = 'no'
result = text1 + ' ' + text2 + ' ' + text3
print(result)`}/>
<Repl code={"'yes or no'"} />

<h3>Repetition (Multiplication)</h3>
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
<Code code={`text = 'What are you doing?'
print('What' in text)
print('How' in text)`} />
<Repl code={`True
False`} />

<h3>Slicing</h3>
<Code code={`text = 'Hello Cruel World!'
print(text[0:5])
print(text[:5])
print(text[6:11])
`} />
<Code code={`text = 'Hello Cruel World!'
print(text[-1])
print(text[-6:])
`} />
<Code code={`text = 'Hello Cruel World!'
print(text[::2])
`} />
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
