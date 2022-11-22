<script>
  import Container from "$lib/Container.svelte";

  let timeline = [
    { time: "2013, 2015", content: "DQN" },
    { time: 2016, content: "Double DQN" },
    { time: 2016, content: "Duelling DQN" },
    { time: 2015, content: "Prioritized Experience Replay" },
    { time: 2016, content: "AC3" },
    { time: 2017, content: "Distributional DQN" },
    { time: 2017, content: "Noisy DQN" },
    { time: 2017, content: "🌈 Rainbow" },
  ];
</script>

<svelte:head>
  <title
    >World4AI | Reinforcement Learning | Value Based Deep Reinforcement Learning</title
  >
  <meta
    name="description"
    content="Value based deep reinforcement learning algorithms like DQN, DDQN, duelling DDQN and so on have achieved state of the art results over the last 10 years."
  />
</svelte:head>

<h1>Value Based Deep Reinforcement Learning</h1>
<div class="separator" />

<Container>
  <p>
    The naive implementation of approximate value based reinforcement learning
    algorithms can lead to divergence, especially when combining non-linear
    function approximators, temporal difference learning and off-policy
    learning. Yet particularly over the last decade researchers have developed
    techniques to reduce the probability of divergence dramatically, making
    off-policy temporal difference learning algorithms the first choice for many
    problemns. This trajectory started with the development of the deep
    Q-network (DQN) by DeepMind. Since then each new iteration of the algorithm
    provided a new improvement, a piece of the puzzle. All of those pieces were
    eventually combined by DeepMind into the so called Rainbow algorithm.
  </p>
  <Container maxWidth="300px">
    <div class="timeline">
      {#each timeline as paper, idx}
        <div
          class="container"
          class:left={idx % 2 === 0}
          class:right={idx % 2 != 0}
        >
          <div class="content">
            <h6 class="title">{paper.time}</h6>
            <p>{paper.content}</p>
          </div>
        </div>
      {/each}
    </div>
  </Container>
  <p>
    In this chapter we are basically going to take a ride down the history lane
    of modern value based deep reinforcement learning. We will start the journey
    by impelementing the DQN algorithm. After that we will cover each subsequent
    improvement separately until we are able to implement a fully featured
    Rainbow algorithm.
  </p>
  <p>
    Before we move on to the discussions and implementations of the individual
    algorithms let us shortly discuss the approach that we are going to take in
    the subsequent chapters. Some of the discussed algorithms built upon the
    previous findings. For example the duelling DQN used the double DQN and not
    the original vanilla DQN as the basis for improvement. In research this
    approach is desirable, as you need to show that your contributions are able
    to improve the current state of the art implementation. We are going to
    cover each of the sections independently. Only the original DQN is going to
    serve as the basis for each of the the improvements. In our opinion this
    makes didactically more sense, as we only need to focus on one piece of the
    puzzle. The combination of the different improvements is going to be
    implemented in the final chapter of this section, the Rainbow algorithm.
  </p>
  <div class="separator" />
</Container>

<style>
  .timeline {
    position: relative;
    margin: 0 auto;
    max-width: 1200px;
  }

  .timeline::after {
    content: "";
    position: absolute;
    width: 3px;
    background-color: var(--text-color);
    top: 0;
    bottom: 0;
    left: 50%;
    margin-left: -3px;
  }

  .container {
    padding: 10px 40px;
    position: relative;
    background-color: inherit;
    border: 1px dashed black;
  }

  .container::after {
    content: "";
    position: absolute;
    width: 20px;
    height: 20px;
    right: -10px;
    background-color: var(--text-color);
    border: 1px solid black;
    top: 9px;
    border-radius: 50%;
    z-index: 1;
  }

  .left {
    left: -50%;
  }

  .right {
    left: 50%;
  }

  .right::after {
    left: -13px;
  }

  h6 {
    font-size: 20px;
  }
</style>
