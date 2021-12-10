<script>
  import Question from "$lib/Question.svelte";
  import Math from "$lib/Math.svelte";
  import Operator from "$lib/Operator.svelte";
  import Repl from "$lib/Repl.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Dictionary Datatype</title>
  <meta
    name="description"
    content="A dictionary in Python is a mutable data type that contains key value pairs."
  />
</svelte:head>

<h1>Dictionary</h1>
<Question>How can we work with dictionaries in Python?</Question>
<div class="separator" />

<h2>Operators</h2>
<h3>Comparisons</h3>
<p>
  When comparison operators <Operator><Math latex={`==`} /></Operator> and <Operator
    ><Math latex={`!=`} /></Operator
  > are applied to dictionaries Python checks if both lists have the same same key
  values pairs.
</p>
<Repl
  code={`>>> my_dict_1 = {'a': 1, 'b': 2}
>>> my_dict_2 = {'a': 1, 'b': 2, 'c': 3}
>>> my_dict_3 = {'b': 2, 'a': 1}
>>> my_dict_1 == my_dict_2
False
>>> my_dict_1 == my_dict_3
True`}
/>
<h3>Membership</h3>
<p>
  The membership operators <Operator>in</Operator> and <Operator
    >not in</Operator
  > determine if a certain key is available in a dictionary.
</p>
<Repl
  code={`>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> 'a' in my_dict
True
>>> 'k' not in my_dict
True`}
/>
<div class="separator" />

<h2>Methods</h2>
<h3>del</h3>
<p>
  del invokes a dictionary method that removes a dictionary element with a
  certain key.
</p>
<Repl
  code={`>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> del my_dict['a']
>>> my_dict
{'b': 2, 'c': 3}`}
/>
<h3>clear</h3>
<p>The clear() method empties the whole dictionary.</p>
<Repl
  code={`>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> my_dict.clear()
>>> my_dict
{}`}
/>
<h3>copy</h3>
<p>The copy() method creates a shallow copy of a dictionary.</p>
<Repl
  code={`>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> my_copy = my_dict.copy()
>>> my_dict['a'] = 22
>>> my_dict
{'a': 22, 'b': 2, 'c': 3}
>>> my_copy
{'a': 1, 'b': 2, 'c': 3}`}
/>
<h3>get</h3>
<p>
  When you try to get a non existing element of a dictionary through a key
  Pyhton will throw an error. The get() method on the other hand will return a
  predetermined value if a key does not exist. If the predetermined value is not
  given to the get() method it will return None in case the key is not contained
  in the dictionary.
</p>
<Repl
  code={`>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> my_dict['d']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'd'
>>> my_dict.get('d')
>>> my_dict.get('d', 'not found')
'not found'`}
/>
<h3>pop</h3>
<p>
  The pop() method removes the object from a dictionary that fits the key and
  returns the removed object.
</p>
<Repl
  code={`>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> removed_value = my_dict.pop('a')
>>> removed_value
1
>>> my_dict
{'b': 2, 'c': 3}`}
/>
<h3>update</h3>
<p>
  The update() method updates a value with the fitting key. If the key is not
  available the new key value pair is created.
</p>
<Repl
  code={`>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> my_dict.update({'d': 4})
>>> my_dict
{'a': 1, 'b': 2, 'c': 3, 'd': 4}
>>> my_dict.update({'a': 99})
>>> my_dict
{'a': 99, 'b': 2, 'c': 3, 'd': 4}`}
/>
<div class="separator" />
