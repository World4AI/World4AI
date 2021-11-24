<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
</script>

<h1>String</h1>
<Question>How can strings be processed in Python?</Question>
<div class="separator"></div>
<p>mention help(int)</p>
<p>string formatting</p>

<h2>Operators</h2>
<Code code={`>>> text1 = 'yes'
>>> text2 = 'or'
>>> text3 = 'no'
>>> text1 + ' ' + text2 + ' ' + text3
'yes or no'
>>>`}/>

<Code code={`>>> '*' * 5
'*****'
>>> 'a' * 3
'aaa'
>>> 'ab' * 2
'abab'
>>>`} />

<Code code={`>>> text = 'What are you doing?'
>>> 'What' in text
True
>>> 'How' in text
False
>>>`} />
<p>
  [] / [1:2]
</p>
<div class="separator"></div>
<h2>Methods</h2>
<div class="separator"></div>
<p>capitalize / casefold / center / count / endswith / find / index / join / lower / lstrip / partition / replace / rindex / rfind / rsplit / rstrip / split / strip / title / upper / zfill  </p>
