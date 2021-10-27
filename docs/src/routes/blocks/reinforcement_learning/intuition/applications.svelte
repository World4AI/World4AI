<script>
    import Grid from '$lib/reinforcement_learning/intuition/applications/Grid.svelte';
    import Pong from '$lib/reinforcement_learning/intuition/applications/Pong.svelte';
    import Go from '$lib/reinforcement_learning/intuition/applications/Go.svelte';
    import Strategy from '$lib/reinforcement_learning/intuition/applications/Strategy.svelte';
    import Finance from '$lib/reinforcement_learning/intuition/applications/Finance.svelte';
    import Robot from '$lib/reinforcement_learning/intuition/applications/Robot.svelte';
</script>


<svelte:head>
    <title>World4AI | Reinforcement Learning Applications</title>
    <meta name="description" content="Reinforcement learning is used in many different applications, like games, finance, robotics, healcare and autonomous vehicles.">
</svelte:head>

<h1>Reinforcement Learning Applications</h1>
<h2>What is reinforcement learning used for?</h2>

<p>It is entirely possible to introduce reinforcement learning using only formal definitions and math, but we will start this journey by exploring what reinforcement learning can actually achieve rather than what it is and how you can apply it. The formalism will come soon enough.</p>

<h3>Games</h3>

<h4>Grid Worlds</h4>

<p>
    Most beginner reinforcement learning problems are grid world problems. They are easy enough to understand and do not require a lot of computational power to solve. A grid world is a rectangular-shaped game with a certain number of rows and columns, where an intersection of a row and a column is a so-called cell in the grid world. A gridworld is (usually) a simple game. You have some sort of a player that can move through the gridworld, some obstacles to prevent the player from entering a certain cell and a goal the player needs to achieve, which then terminates (or restarts) the game. But of course there are grid worlds that are substantially more complex. These can for example include powerups, enemies and many levels.
</p>

<div class="flex-center">
    <Grid />
</div>

<p>
    In the above example the player has to move the circle from the bottom left corner to the upper left corner. This seems like a trivial task for a human being, but it gets nontrivial if you are not allowed to hardcode the solution. Instead you have to make your computer learn the goal of the game and the strategy to achieve the goal. That is exactly where reinforcement learning comes into play. By applying reinforcement learning algorithms it becomes possible to learn the optimal behaviour, where the circle arrives at the goal in as few steps as possible (as indicated by the arrows in the cells).
</p>

<h4>Atari 2600 Games</h4>

<p>
    Computer games have become a testing ground for reinforcement learning algorithms. Most new algorithms are tested on the Atari 2600 games in order to show how efficient the algorithms are. For a human it is not especially hard to learn the rules of the game (although it might require some time to master the game), but for computers it is an entirely different story. Due to the vast number of possible pixel values on the screen the usual strategies that are used to solve the grid worlds break down. Good solutions require the use of neural networks. 
</p>

<p>
    The most known atari game is probably pong. In pong for example you steer a paddle at the right of the screen. The goal of the game is to prevent the ball from reaching the right border by moving the paddle into the path of the ball. The ball bounces off the paddle and if the ball to passes the paddle of the enemy, then we have earned ourselve a point. The enemy on the left can be controlled by a simple hardcoded AI or just a different player. In reinfocement learning a trained AI is responsible for the control of the right paddle.
</p>

<div class="flex-center">
    <Pong />
</div>

<p>
    The computer receives the current frames of the game and has to decide how to act based on the pixel values. There are of course versions of the game, where the computer receives the positions of the ball and the paddles as coordinate values, but using that version would be essentially cheating. Looking at the pictures and behaving accordingly is not unlike how humans act. Therefore it is especially impressive that with the help of reinforcement learning it is possible to create computer programs that are able to beat human scores in all the Atari 2600 games while making decisions on the basis of the screenshots of the game. 
</p>

<h4>Board Games</h4>

<p>
    Board games, like backgammon, chess and go used to be the frontier for AI. There was an assumption that a computer would require creativity and imagination to beat a professional player at backgammon, chess and go. Essentially that meant that the computer needed to possess human characteristics in order to win against a professional player. Nevertheless in all three games professionals and even world champions were beaten by AI systems.
</p>

<p>
    Even though all solutions were an unbelievable milestone for the AI community, the win of DeepMind with their AlphaGo system against the go world champion was most likely the most impressive and the hardest achievement in AI thus far. For a number of years the challenge of winning against the world champion was considered impossible. The number of legal board positions in the game of go is far greater than there are atoms in the observable universe. Iterating through all positions is therefore impossible. Nevertheless, not only did the algorithm win against the world champion Lee Sedol in the 4 of 5 games, but according to go experts AlphaGo showed creativity. In the second of the five games AlphaGo shocked the world with the now iconic move. This move has become known as <strong>Move 37</strong>.
</p>

<div class="flex-center">
    <Go />
</div>

<h4>Modern Games</h4>

<p>
    Compared to Atari games, modern games have become a lot more complex and even for human players there is a steep learning curve, especially if you wish to become a professional player. In spite of that OpenAI and DeepMind have beaten top players in two of the most famous esports games, Dota II and StarCraft II.
    StarCraft II for example is a so-called rts (real time strategy) game. In these types of games you have to collect resources, build workers and different types of attack vehicles, construct buildings, improve your technology, scout the area for opponents and to finally destroy the buildings and units of your opponents. Many of these decisions have tradeoffs and there is no single best strategy, the player has to adapt to the current situation.
</p>

<div class="flex-center">
    <Strategy />
</div>

<p>
    The image above shows how an imagined rts game configuration might look like. The picture indicates how a player has to balance several decisions at the same time. What makes these types of games additionally hard is the so-called “fog of war”. The grey areas are not visible to the player, so that only a part of the map is observable. That creates an information imbalance and requires scouting the area to gain information.
</p>

<h3>Finance</h3>

<p>
    Nowadays it seems that reinforcement learning is taking over the financial industry in every aspect imaginable. From valuing financial products to chatbots that communicate with prospective clients. 
</p>

<div class="flex-center">
    <Finance />
</div>

<p>
    The most exciting part still seems to be portfolio management though. Imagine an artificial intelligence that decides what financial instrument to invest in and in what proportion. And the exact decisions could depend on the risk aversion of the client. There is an abundance of financial data (often free) going back sometimes a hundred of years. That data can be used to train a reinforcement learning AI to potentially perform better than humans over a long period of time. Even if the AI performs equally well to human portfolio managers, the banks could cut costs, as automated trading bots tend to be much cheaper than human portfolio managers. 
</p>

<h3>Robotics</h3>

<p>The field of robotics is vast. We could talk about robots on assembly lines, drones or bipedal robots. In all the above mentioned cases it is possible to apply reinforcement learning to learn the desired task for the robot.</p>

<div class="flex-center">
    <Robot />
</div>

<p>
    For example a bipedal robot can be taught to walk on two legs through the means of reinforcement learning. Each good or bad step of the robot can be used as a learning experience. In fact that is partially already actively done but still remains an active research field. At the moment most successful bipedal robots, like those made by Boston Dynamics, are not actually trained through reinforcement learning, but are hardcoded to solve their task. Therefore these are great feats of pure engineering and not AI, but 10 years down the road and AI will probably replace a lot of hardcoded parts. 
</p>

<h3>Healthcare</h3>

<p>
    The possible advances in healthcare through the help of reinforcement learning are probably the ones that are of the greatest benefit to the general population. Several applications come to mind when I think about the intersection of healthcare and reinforcement learning.
</p>

<p>
    Individual treatment plans for example, where drugs and treatments are adjusted quickly depending on the reaction of the patient could be developed. With each new patient the experience of the AI would increase. Unlike human doctors that have to go through medical studies, practice and have to eventually retire, the experience of AI could be centralized and made available to newer systems without loss of data. Thus the amount of mistakes and the cost of treatment should potentially decrease.
</p> 

<p>
    The development of new drugs is another exciting possible application of reinforcement learning. The abundance of medical data should allow the AI to learn to accelerate the drug discovery phase while making the overall development process safer. 
</p>

<p>To my knowledge both applications are experimental and are not fully implemented yet, but research in that area has been increasing.</p>

<h3>Autonomous Vehicles</h3>

<p>
    Autonomous vehicles (a.k.a. self-driving cars) are at the moment of writing the current frontier for reinforcement learning. There are many car companies that invest in self-driving cars. Newer car companies like Tesla and Google’s Waymo and old German car manufacturers like Volkswagen all invest an enormous amount of time and money in the development of autonomous vehicles. Research in the area has been going on since at least the 80’s, with demonstrations by Daimler, but the behaviour of these vehicles in edge cases made their use often dangerous for everyday use. Since the DARPA Grand Challenge (2007) great leaps have been made and reinforcement learning played a huge role in that success story. It is probably only a matter of time until autonomous vehicles become fully legalized. The advantages are tremendous. Expected lower mortality rates due to traffic accidents, lower cost for taxi fares and more efficient logistics, as drivers won't have to sleep on overnight journeys.
</p> 

<style >
    .flex-center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>