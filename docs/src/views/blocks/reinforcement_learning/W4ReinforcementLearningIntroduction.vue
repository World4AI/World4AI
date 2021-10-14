<template>
    <main>
        <div class="container">
            <h1>Introduction</h1>
            <h2>What is reinforcement learning?</h2>
            <p>A reasonable way to think about reinforcement learning is to think about animals or even children whom you would like to teach good behaviour and discourage from bad behaviour. The way you would go about it would probably involve some reward in case the animal (let’s say a dog)  behaves the way you desire and punishment if it doesn’t. If you would like to teach the dog to sit down when you blow a whistle, then blow the whistle until the dog sits down. When it does, give it a treat. If you keep doing that the dog should learn sooner or later to sit down at the sound of the whistle. The process would most likely take time and require a lot of patience, but animals are quite good at learning the right behaviour when it comes to acquiring food.</p>
            <p>In a later chapter we are going to introduce a more formal definition of reinforcement learning, but keep the above metaphor in mind. This intuition will guide you quite nicely in your journey through reinforcement learning.</p>
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

    p {
        grid-column-start: 3;
        line-height: 2;
        font-size: 18px;
    }

</style>