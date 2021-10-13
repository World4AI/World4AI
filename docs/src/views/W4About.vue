<template>
    <main>
        <div class="container">
            <h1>About</h1>
            <h2>What is World4AI?</h2>

            <p>World4AI is my answer to the question “How can I get to an intermediary level in AI/Machine Learning?”.</p>

            <p>The main purpose of Wordl4AI is for me to have a personal platform/blog that I can utilize to learn AI. The attempts that I made so far to teach myself Machine Learning were not especially successful, but there are several reasons why I think that a blog might give me the necessary push to pass the beginner level in AI.</p>

            <h3>Improvement over time</h3>
            <p>The written word is easy to change and if I make a mistake or find a better way to explain a topic I can change the content, learn from the mistake and move on.</p>

            <h3>Pressure</h3>
            <p>I found that pressure, either in the form of the wish for good grades or through deadlines in the workplace put me in a state of high focus. By committing to at least one article a week I should be able to put enough pressure on me to keep going.</p>

            <h3>Learning by teaching</h3>
            <p>I am a firm believer in the “Learning by Teaching” method. By preparing materials for others, I have to ask myself questions from the perspective of a reader and if I am not able to answer those myself, then I have to redo the article, until I have captured the essence of the topic.</p>

            <h3>Progress tracking</h3>
            <p>So far I have had difficulties staying on track. The topics that I learned were all over the place and at the end I had not a lot to show for. This platform will allow me to track my progress and to focus on the topics that I have not mastered yet.</p>

            <h3>A better form of CV</h3>
            <p>At the moment I am considering switching careers from finance to AI. If the goal of me learning AI is to find a new job in the field of machine learning, then I can not imagine a better form of CV than a personalized website showing off my progress and commitment.</p>

            <h2>Teaching and Learning Methods</h2>

            <h3>Intuition</h3>
            <p>Especially as a beginner it is essential to build up at least some intuition before jumping into the underlying math. The math is usually easier to digest once you understand the core ideas of the algorithm. To make the transition to math easier I will use graphs and metaphors to explain the hard parts of the algorithms. This is also the part that should hopefully stick for a very long time. The mathematical proofs and the different types of implementations are easily forgotten once you stop practicing, but the general idea of why and how should become second nature to me.</p>

            <h3>Math</h3>
            <p>A lot of AI is math, but math should not stop anyone from learning artificial intelligence. In my experience math is not as complex, as the convoluted notation suggests. Being able to read the notation is sometimes half the battle. Furthermore, in order to be able to produce a really good explanation on an intuitive level, sometimes I have to fight through the math first.</p>            
 
            <h3>Programming</h3>
            <p>For me personally programming is the most exciting part of machine learning. It is the part that produces actual results. At this stage you are able to see how your program is able to classify digits or to beat a computer game. Seeing how the neural network improves can only be described as magical. 
            It is usually at this stage where you have your “aha moments”. The math becomes visible and understandable in a very convenient way and you jump between the text book and the computer editor to appreciate the beauty of the algorithm. </p>
            
        </div>
    </main>
</template>

<script>
export default {

}
</script>

<style scoped>
    .container {
        display: grid;
        grid-template-columns: 1fr 1fr 6fr 1fr 1fr;
        grid-template-rows: auto;
        gap: var(--gap);
        justify-content: center;
        align-items: center;
    }

    h1 {
        color: var(--text-color);
        text-transform: uppercase;
        font-weight: 700;
        grid-column-start: 3;
        align-self: center;
        justify-self: left;
        font-size: 70px;
        border-top: 1px solid var(--text-color);
        border-bottom: 1px solid var(--text-color);
    }

    h2 {
        color: var(--text-color);
        grid-column-start: 3;
        text-transform: uppercase;
        font-size: 30px;
        font-weight: 400;
        letter-spacing: 5px;
    }

        h3 {
        color: var(--text-color);
        grid-column-start: 3;
        text-transform: uppercase;
        font-size: 25px;
        font-weight: 400;
        letter-spacing: 5px;
    }

    p {
        grid-column-start: 3;
        line-height: 2;
        font-size: 18px;
    }

    .info {
        box-shadow: -5px 5px 0px 0px var(--main-color-1), 5px -5px 0px 0px var(--main-color-1);
        padding: 10px;
    }


</style>