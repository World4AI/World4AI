import { Agent } from "$lib/reinforcement_learning/common/Agent";

class RandomAgent extends Agent {
    constructor(observationSpace, actionSpace) {
        super(observationSpace, actionSpace)
    }

    act(observation) {
        return Math.floor(Math.random() * this.actionSpace.length);
    }

    arrows() {
        let arrows = [];
        
    }
}

export { RandomAgent }