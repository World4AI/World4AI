============
Applications
============

It is entirely possible to introduce reinforcement learning using only formal definitions and math, but I would like to start this journey by exploring what reinforcement learning can actually achieve rather than what it is and how you can apply it. The formalism will come soon enough.

Games
-----

Grid Worlds
===========

Most beginner reinforcement learning problems are grid world problems. They are easy enough to understand and do not require a lot of computational power to solve.
A grid world is a rectangular-shaped game with a certain number of rows and columns, where an intersection of a row and a column is a so-called cell in the grid world. A gridworld is (usually) a simple game. You have some sort of a player that can move through the gridworld, some obstacles to prevent the player from entering a certain cell and a goal the player needs to achieve, which then terminates (or restarts) the game. But of course there are grid worlds that are substantially more complex. These can for example include powerups, enemies and many levels.

.. image:: ../_static/images/reinforcement_learning/applications/grid_world.svg
   :align: center

Atari 2600 Games
================

Computer games have become a testing ground for reinforcement learning algorithms. Most new algorithms are tested on the Atari 2600 games in order to show how efficient the algorithms are. For a human it is not especially hard to learn the rules of the game (although it might require some time to master the game), but for computers it is an entirely different story.

.. image:: ../_static/images/reinforcement_learning/applications/breakout.svg
   :align: center


The most known atari games are probably pong and breakout. In breakout for example you steer the paddle at the bottom of the screen. The goal of the game is to prevent the ball from falling by moving the paddle in the path of the ball. The ball bounces off the paddle and if it touches one of the blocks at the top of the screen, the block disappears, the ball starts moving in the opposite direction and you get some points. The game ends when either the ball falls on the ground or when you have destroyed all the blocks.

The computer receives the current frames of the game and has to decide how to act based just on the pixel values. There are of course versions of the game, where the computer receives the positions of the ball, the paddle and the boxes as coordinate values, but using that version would be essentially cheating. Looking at the pictures and behaving accordingly is not unlike how humans act. Therefore it is especially impressive that with the help of reinforcement learning it is possible to create computer programs that are able to beat human scores in all the Atari 2600 games while making decisions on the basis of the screenshots of the game.
   
   

Board Games
===========

Board games, like backgammon, chess and go used to be the frontier for ai. There was an assumption that a computer would require creativity and imagination to beat a professional player at backgammon, chess and go. Essentially that meant that the computer needed to possess human characteristics in order to win against a professional player. Nevertheless in all three games professionals and even world champions were beaten by AI systems.

Even though all solutions were an unbelievable milestone for the AI community, the win of DeepMind with their AlphaGo system against the world champion was the most impressive for me. For a number of years the challenge of winning against the world champion was considered impossible. The number of legal board positions in the game of go is far greater than there are atoms in the observable universe. Iterating through all positions is therefore impossible. Nevertheless, not only did the algorithm win against the world champion Lee Sedol in the 4 of 5 games, but according to go experts AlphaGo showed creativity. In the second of the five games AlphaGo shocked the world with the now iconic move. This move has become known as “Move 37”.

.. image:: ../_static/images/reinforcement_learning/applications/game_of_go.svg
   :align: center


Modern Games
============

Compared to Atari games, modern games have become a lot more complex and even for human players there is a steep learning curve, especially if you wish to become a professional player. In spite of that OpenAI and DeepMind have beaten top players in two of the most famous esports games, Dota II and StarCraft II.
StarCraft II for example is a so-called rts (real time strategy) game. In these types of games you have to collect resources, build workers and different types of attack vehicles, construct buildings, improve your technology, scout the area for opponents and to finally destroy the buildings and units of your opponents. Many of these decisions have tradeoffs and there is no single best strategy, the player has to adapt to the current situation.

.. image:: ../_static/images/reinforcement_learning/applications/rts.svg
   :align: center

The image above shows how an imagined rts game configuration might look like. The picture indicates how a player has to balance several decisions at the same time. What makes these types of games additionally hard is the so-called “fog of war”. The grey areas are not visible to the player, so that only a part of the map is observable. That creates an information imbalance and requires scouting the area to gain information.

Trading
-------

Robotics
--------

Medicine
--------

Reccomender Systems
-------------------

The Frontiers
-------------

