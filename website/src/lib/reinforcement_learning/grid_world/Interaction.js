import { onMount } from "svelte";
import { writable } from 'svelte/store';

class Interaction{
  constructor(agent, env, fps=5) {
    this.agent = agent;
    this.env = env;
    //var for fps calculation
    this.fpsInterval = 1000/fps;
    //writable stores
    this.actionStore = writable(null);
    this.observationStore = writable(null);
    this.rewardStore = writable(null)

    onMount(() => {
      this.reset();
      this.then = Date.now();
      let frame;

      let loop = () => {
        frame = requestAnimationFrame(loop);
        let now = Date.now();
        let elapsed = now - this.then;
        if (elapsed >= this.fpsInterval){
          this.then = now;
          this.interact();
        }
        return () => {
        cancelAnimationFrame(frame);
        };
      }
  
      requestAnimationFrame(loop)
  
    })
  }

  reset() {
    this.observation = this.env.reset();
    this.action = null;
    this.reward = null;
    this.payload = {};

    this.observationStore.set(this.observation);
    this.rewardStore.set(this.reward);
    this.actionStore.set(this.action);
  }

  interact() {
    // reset all variables
    if (this.payload.done) {
      this.reset();
    } else {
      this.action = this.agent.act(this.observation);
      this.actionStore.set(this.action);
      this.payload = this.env.step(this.action);
      this.observation = this.payload.observation;
      this.reward = this.payload.reward;
      this.observationStore.set(this.observation);
      this.rewardStore.set(this.reward);
    }
  }
}

export {Interaction}
