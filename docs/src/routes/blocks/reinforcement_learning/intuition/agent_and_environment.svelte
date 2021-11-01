<script>
    import { RandomAgent } from '$lib/reinforcement_learning/common/RandomAgent';
    import { GridEnvironment } from '$lib/reinforcement_learning/common/GridEnvironment';
    import { gridMap } from '$lib/reinforcement_learning/common/maps';

    import Grid from '$lib/reinforcement_learning/intuition/applications/Grid.svelte';
    import AgentEnvironment from '$lib/reinforcement_learning/intuition/agent_and_environment/AgentEnvironment.svelte';
    import Robot from '$lib/reinforcement_learning/intuition/agent_and_environment/Robot.svelte';


    let env_1 = new GridEnvironment(gridMap);
    let agent_1 = new RandomAgent(env_1.getObservationSpace(), env_1.getActionSpace());
    
    let env_2 = new GridEnvironment(gridMap);
    let agent_2 = new RandomAgent(env_2.getObservationSpace(), env_2.getActionSpace());   

</script>

<svelte:head>
    <title>World4AI | Reinforcement Learning | Agent and Environment</title>
    <meta name="description" content="The agent and the environment are the two main components in reinforcement learning. The agent is the component that generates the decisons. The environment is everything else.">
</svelte:head>

<h1>Agent and Environment</h1>

<h2>What are the main components of reinforcement learning?</h2>

<p>All of reinforcement learning is based on two main components, the <strong>agent</strong> and the <strong>environment</strong>. To introduce both components we will follow the customary route taken in the reinforcement learning education and use a grid world example.</p>  

<div class="flex-center">
    <Grid env={env_1} agent={agent_1} />
</div>

<p>The gridworld above is the same that we have seen in the previous section. We are going to use this simple game throughout the coming sections to get familiar with the basics of reinforcement learning.</p> 

<p>In the gridworld the main player is represented by a circle. The player can move into 4 different directions: North, East, South and West. If the circle is against a wall (outermost lines) or a barrier (boxes in the middle of the grid) the circle can not move in that particular direction. The goal of the game is for the circle to reach the triangle in the bottom left corner in as few steps as possible, but at this point in time the circle moves randomly.</p>

<p>Intuitively we could say that the circle is the agent and the grid world is the environment the agent interacts with, but that definition would not be entirely correct. There is actually a relatively strict separation between the agent and the environment.</p>

<p>All the agent can actually do is to make the decisions. In the case of the above gridworld the agent chooses the direction, meaning north, west, east or south. Whether the circle actually moves in that direction is outside of the influence of the agent.</p>

<p class="info">The agent is the code that makes the decisions. Do not mix up the agent with its physical / graphical representation in the environment.</p>

<div class="flex-center">
    <Grid env={env_2} agent={agent_2} showArrows=true/>
</div>

<p>For example in the gridworld above the agent can decide to go north even when the circle is against a barrier or a wall in the northern direction. That decision is legitimate in many grid worlds, but the position of the circle will not change. The arrow indicates the choice the agent made when it was in the previous cell. From time to time we can observe that the agent tries to move against a wall or a barrier, but the grid world does not react to that decision.</p>  

<p>The agent is the program that generates the decision and the decision of the agent is then relayed to the environment. The environment processes the decision, but that is not something that the agent can influence. In the simple grid-world game if the agent decides to go north the circle might actually move north or it could move in a totally different direction or not move at all.</p>

<p>The environment on the other hand is everything that is not the agent. The enviroment in the grid world reacts to the decisions of the agent and decides what position the circle moves to. Additionally the environment rewards the agent for its decisions.</p>

<p class="info">Anything outside of the agent is the environment.</p>

<div class="flex-center">
    <Robot />
</div>
<p>A different common example that is often used to make the distinction between the agent and the environment is that of a robot that interacts with the real world. The goal of the agent might be to move the robot from right to left to recharge the battery. The agent (represented by a chip on the right) sends the decison to move the robot. But it is entirely possible that the robot does not even start moving because the battery is already completely depleted. In this example the agent is the code that makes the decision to move left. While the arms, the legs, the cooling system of the robot, the battery, the floor and everything else in the image is part of the environment.</p>

<p>A similar argument can be made for a human or any other form of a biological machine. The neural network in our brain makes the decisions to move, to study or to sleep. How the body actually reacts is not really in our control. For example the movement could be stopped when you are parallyzed either because of an illness or through fear. The desire to study could be stopped through external triggers like the smell of food or a habit to listen to music. And the decision to go to bed to fall asleep is not always accompanied by actual sleep. We do not have full control of our body. For example, no matter how hard we try, we can not stop our heart though our will.</p>

<div class="flex-center">
    <AgentEnvironment />
</div>

<p>
    Before we move on to the next section let us clarify the visual notation we are going to use in the reinforcement learning block. The grid on the left is going to depict the environment. You can think about a grid world or about a chess/go board. The symbol on the right is going to represent the agent. Think about a chip that runs on the computer and produces the decisions that are then transmitted to the environment. Finally, the lines represent the interaction between the two components.  
</p>
