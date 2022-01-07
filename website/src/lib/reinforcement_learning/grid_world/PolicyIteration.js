import { writable } from 'svelte/store';

class PolicyIteration{
  constructor(observationSpace, actionSpace, model, theta, gamma) {
    this.observationSpace = observationSpace;
    this.actionSpace = actionSpace;
    this.policy = {}; 
    this.valueFunction = {};
    this.model = model;
    this.theta = theta;
    this.gamma = gamma;
    this.maxDelta = 0;

    //create random policy
    observationSpace.forEach((obs, idx) => {
      if (!this.policy[obs.r]) {
        this.policy[obs.r] = {};
      }
      this.policy[obs.r][obs.c] = this.randomChoice(actionSpace);
    });

    //init 0 state value function
    observationSpace.forEach((obs, idx) => {
      if (!this.valueFunction[obs.r]) {
        this.valueFunction[obs.r] = {};
      }
      this.valueFunction[obs.r][obs.c] = 0;
    });

    this.policyStore = writable(this.policy);
    this.valueStore = writable(this.valueFunction);
  }

  policyEvaluation() {
    do  {
      this.policyEvaluationStep();
    } while (this.maxDelta > this.theta)
  }

  policyEvaluationStep() {
    let oldValueFunction = JSON.parse(JSON.stringify(this.valueFunction));
    let newValueFunction = JSON.parse(JSON.stringify(this.valueFunction));
    this.maxDelta = 0;
    
    this.observationSpace.forEach((obs) => {
      let action = this.policy[obs.r][obs.c];
      let v = 0;
      this.model[obs.r][obs.c][action].forEach((data) => {
        let prob = data.probability;
        let reward = data.reward;
        let nextObs = data.observation;
        let done;
        if(data.done === true){
          done = 0;
        } else {
          done = 1;
        }
        v += prob*(reward + this.gamma * oldValueFunction[nextObs.r][nextObs.c] * done);
      });
      newValueFunction[obs.r][obs.c] = v;
      let delta = Math.abs(v - oldValueFunction[obs.r][obs.c]); 
      if(delta > this.maxDelta){
        this.maxDelta= delta;
      }
    });
    this.valueFunction = newValueFunction;
    this.valueStore.set(this.valueFunction);
  }

  randomChoice(arr) {
    return arr[Math.floor(arr.length * Math.random())];
  }

  policyImprovement() {
    let newPolicy = JSON.parse(JSON.stringify(this.policy));
    this.observationSpace.forEach((obs) => {
      let vmax = -1000000;
      let argmax = 0;

      this.actionSpace.forEach((action) => {
        let v = 0;
        this.model[obs.r][obs.c][action].forEach((data) => {
          let prob = data.probability;
          let reward = data.reward;
          let nextObs = data.observation;
          let done;
          if(data.done === true){
            done = 0;
          } else {
            done = 1;
          }
          v += prob*(reward + this.gamma * this.valueFunction[nextObs.r][nextObs.c] * done);
        });
        if (v > vmax) {
          vmax = v;
          argmax = action;
        }
      });
      newPolicy[obs.r][obs.c] = argmax;
    });
    return newPolicy;
  }
  
  policyIteration() {
    while(true) {
      this.policyEvaluation();
      let newPolicy = this.policyImprovement();
      if (JSON.stringify(newPolicy) ==JSON.stringify(this.policy)) {
        break;
      }
      this.policy = newPolicy;
      this.policyStore.set(this.policy);
    }
  }
}

export {PolicyIteration}
