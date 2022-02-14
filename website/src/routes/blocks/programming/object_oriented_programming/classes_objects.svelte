<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Code from "$lib/Code.svelte";
  import Repl from "$lib/Repl.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Objects and Classes</title>
  <meta
    name="description"
    content="Classes are the blueprints for objects. We use classes to define the general structure of each instance of that class."
  />
</svelte:head>

<h1>Classes and Objects</h1>
<Question
  >What is the difference between classes and objects and how can you create
  those in Python?</Question
>
<div class="separator" />

<p class="info">
  A class is a blueprint from which we can construct objects that share a common
  predetermined structure.
</p>
<p>
  If we want to create an object, we have to create a blueprint from which each
  of those objects can be generated. The class determines the common data and
  behaviour that objects of the same class will share.
</p>
<svg version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="TriangleOutL" overflow="visible" orient="auto">
      <path
        transform="scale(.8)"
        d="m5.77 0-8.65 5v-10l8.65 5z"
        fill="var(--text-color)"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g stroke="var(--text-color)">
    <g id="background" fill="none" stroke-width="0.3px" stroke-dasharray="1,2">
      <path d="m20 50 20-20" />
      <path d="m20 60 30-30" />
      <path d="m20 70 40-40" />
      <path d="m20 80 50-50" />
      <path d="m20 90 60-60" />
      <path d="m20 100 70-70" />
      <path d="m20 110 80-80" />
      <path d="m20 120 90-90" />
      <path d="m20 130 100-100" />
      <path d="m20 140 110-110" />
      <path d="m20 150 120-120" />
      <path d="m20 160 130-130" />
      <path d="m20 170 140-140" />
      <path d="m20 180 150-150" />
      <path d="m20 190 160-160" />
      <path d="m20 200 170-170" />
      <path d="m20 210 180-180" />
      <path d="m20 220 190-190" />
      <path d="m20 230 200-200" />
      <path d="m20 240 210-210" />
      <path d="m20 250 220-220" />
      <path d="m20 260 230-230" />
      <path d="m20 270 240-240" />
      <path d="m20 280 250-250" />
      <path d="m20 290 260-260" />
      <path d="m20 300 270-270" />
      <path d="m20 310 280-280" />
      <path d="m20 320 290-290" />
      <path d="m20 330 300-300" />
      <path d="m20 340 310-310" />
      <path d="m20 350 320-320" />
      <path d="m20 360 330-330" />
      <path d="m20 370 340-340" />
      <path d="m20 380 350-350" />
      <path d="m20 390 360-360" />
      <path d="m20 400 370-370" />
      <path d="m20 410 380-380" />
      <path d="m20 420 390-390" />
      <path d="m20 430 400-400" />
      <path d="m20 440 410-410" />
      <path d="m20 450 420-420" />
      <path d="m20 460 430-430" />
      <path d="m20 470 440-440" />
      <path d="m20 480 450-450" />
      <path d="m30 480 440-440" />
      <path d="m40 480 430-430" />
      <path d="m50 480 420-420" />
      <path d="m60 480 410-410" />
      <path d="m70 480 400-400" />
      <path d="m80 480 390-390" />
      <path d="m90 480 380-380" />
      <path d="m100 480 370-370" />
      <path d="m110 480 360-360" />
      <path d="m120 480 350-350" />
      <path d="m130 480 340-340" />
      <path d="m140 480 330-330" />
      <path d="m150 480 320-320" />
      <path d="m160 480 310-310" />
      <path d="m170 480 300-300" />
      <path d="m180 480 290-290" />
      <path d="m190 480 280-280" />
      <path d="m200 480 270-270" />
      <path d="m210 480 260-260" />
      <path d="m220 480 250-250" />
      <path d="m230 480 240-240" />
      <path d="m240 480 230-230" />
      <path d="m250 480 220-220" />
      <path d="m260 480 210-210" />
      <path d="m270 480 200-200" />
      <path d="m280 480 190-190" />
      <path d="m290 480 180-180" />
      <path d="m300 480 170-170" />
      <path d="m310 480 160-160" />
      <path d="m320 480 150-150" />
      <path d="m330 480 140-140" />
      <path d="m340 480 130-130" />
      <path d="m350 480 120-120" />
      <path d="m360 480 110-110" />
      <path d="m370 480 100-100" />
      <path d="m380 480 90-90" />
      <path d="m390 480 80-80" />
      <path d="m400 480 70-70" />
      <path d="m410 480 60-60" />
      <path d="m420 480 50-50" />
      <path d="m430 480 40-40" />
      <path d="m440 480 30-30" />
      <path d="m450 480 20-20" />
      <path d="m460 480 10-10" />
      <path d="m20 40 10-10" />
    </g>
    <g id="foreground">
      <g fill="none">
        <path d="m20 500v-480" marker-end="url(#TriangleOutL)" />
        <path
          d="m0 480h480"
          marker-end="url(#TriangleOutL)"
          stroke-width="1px"
        />
        <path
          id="horizontal-coordinate"
          d="m20 120h460"
          stroke-dasharray="8, 4"
        />
        <path
          id="vertical-coordinate"
          d="m220 480v-460"
          stroke-dasharray="8, 4"
        />
      </g>
      <circle
        id="coordinates"
        cx="220"
        cy="120"
        r="10"
        fill="var(--text-color)"
      />
    </g>
  </g>
</svg>
<p>
  The class "Coordinates" for example is a blueprint for an object that has an
  x-coordinate, a y-coordinate and several methods that provide an interface to
  change those coordinates.
</p>
<h2>Arguments</h2>
<p>Below is the simplest implementation of the class "Coordinates".</p>
<Code
  code={`
class Coordinates:
    pass    

coordinates = Coordinates()
coordinates.x = 10
coordinates.y = 5

print(f'The x coordinate is {coordinates.x}')
print(f'The y coordinate is {coordinates.y}')
  `}
/>

<p>
  In Python we create a class by using the <Highlight>class</Highlight> keyword followed
  by the name of the class. It is good practice to use CapWords for class names.
  Every attribute (data) and method (functionality) of the class has to be contained
  within the indented block below the class keyword. For now we use only the <Highlight
    >pass</Highlight
  > statement to indicate that there are no attributes or methods associated with
  that class.
</p>
<p>
  To create an object, also called an instance of a class, we call the class
  like we would call a function by using the name of the class followed by round
  brackets. The result of that operation is assigned to a variable. That is what
  happens when we write
  <Highlight>coordinates = Coordinates()</Highlight>. We create an object from
  the "Coordinates" blueprint and use the "coordinates" varaible to reference
  that object.
</p>
<p>
  We use the dot operator <Highlight>.</Highlight> to create and access the attributes
  and methods of an object. When we use <Highlight>coordinates.x = 10</Highlight
  > for example we generate the attribute x with the associated value 10.
</p>

<p>
  The way we defined and used classes above is valid, but not efficient. Usually
  we want to identify the necessary attributes of the class when we define the
  class and not once the object is already constructed. For example we know from
  the very start that each "Coordinates" object should have the x and y
  coordinate and we could determine those values when we construct the object.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y): 
        self.x = x
        self.y = y

coordinates = Coordinates(10, 5)
  `}
/>
<p>
  Above we use the <Highlight>__init__()</Highlight> method. Methods that start and
  end with a double underscores are called magic or dunder (short for double underscore)
  methods. Those methods usually run automatically when a certain process is triggered.
  The "__init__" method for example runs every time a new object is constructed.
  This method is often called init-method or constructor method.
</p>
<p>
  In the above example the method expects three arguments. The x-coordinate, the
  y-coordinate and <Highlight>self</Highlight>, which refers to a particular
  instance of that class. The __init__ method is run when the object is created,
  therefore when we write <Highlight>Coordinates(10, 5)</Highlight> the arguments
  in the brackets actually refer to the x and y parameters. The self parameter automatically
  receives the actual object that calls the __init__ method, the coordinates object
  is passed as the parameter.
</p>
<p>
  Below we demonstrate the same process with a different example. Instead of
  using the __init__ method that is automatically called when the object is
  created we define a custom init function. We call the function after the
  object is created and use the object as an input into that function. In the
  below example it is more obvious what the self parameter is replaced by.
</p>
<Code
  code={`
class Coordinates:
    pass

def init(self, x, y):
    self.x = x
    self.y = y

coordinates = Coordinates()
init(coordinates, 10, 5)
  `}
/>
<div class="separator" />

<h2>Methods</h2>
<p>
  Methods are functions that are defined at the level of classes and are usually
  expected work with and on attributes of those classes. Below is a
  demonstration of an implementation of a method that prints the x and y
  coordinates. Methods receive the <Highlight>self</Highlight> argument and work
  therefore with the data of a particular instance. To call the method on the instance
  we use the
  <Highlight>.</Highlight> operator followed by the name of the method.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print_coordinates(self):
        print(f'The x coordinate is: {self.x}')
        print(f'The y coordinate is: {self.y}')

coordinates = Coordinates(10, 5)
coordinates.print_coordinates()
  `}
/>
<Repl
  code={`
The x coordinate is: 10
The y coordinate is: 5
  `}
/>
<div class="separator" />

<style>
  svg {
    max-width: 400px;
  }
</style>
