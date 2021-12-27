import { onMount } from "svelte";

function interact(agent, env, fps=5){
  let observation;
  let action;
  let reward;
  let payload;
  //vars for fps calculation
  let then;
  let now;
  const fpsInterval = 1000/fps;

  function reset() {
    observation = env.reset();
    action = null;
    reward = null;
    payload = {};
    then = Date.now();
  }

  function interact() {
    // reset all variables
    if (payload.done) {
      reset();
    } else {
      action = agent.act(observation);
      payload = env.step(action);
      observation = payload.observation;
    }
  }

  onMount(() => {
    reset();
    let frame;

    requestAnimationFrame(function loop() {
      frame = requestAnimationFrame(loop);
      now = Date.now();
      let elapsed = now - then;
      if (elapsed >= fpsInterval){
        then = now;
        interact();
      }

    });

    return () => {
      cancelAnimationFrame(frame);
    };
  })}

export {interact}
