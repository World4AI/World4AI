import { Agent } from "$lib/reinforcement_learning/common/Agent";

class DeterministicAgent extends Agent {
    constructor(observationSpace, actionSpace) {
        super(observationSpace, actionSpace)
    }

    act(observation) {
        if(observation.r === 0 && observation.c === 0) {
            return 2;
        }
        else if(observation.r === 0 && observation.c === 1) {
            return 2;
        }
        else if(observation.r === 0 && observation.c === 2) {
            return 2;
        }
        else if(observation.r === 0 && observation.c === 3) {
            return 2;
        }
        else if(observation.r === 0 && observation.c === 4) {
            return 2;
        }
        else if(observation.r === 1 && observation.c === 0) {
            return 1;
        }
        else if(observation.r === 1 && observation.c === 1) {
            return 1;
        }
        else if(observation.r === 1 && observation.c === 2) {
            return 1;
        }
        else if(observation.r === 1 && observation.c === 3) {
            return 2;
        }
        else if(observation.r === 1 && observation.c === 4) {
            return 2;
        }
        else if(observation.r === 2 && observation.c === 3) {
            return 2;
        }
        else if(observation.r === 2 && observation.c === 4) {
            return 2;
        }
        else if(observation.r === 3 && observation.c === 4) {
            return 3;
        }
        else if(observation.r === 3 && observation.c === 3) {
            return 3;
        }
        else if(observation.r === 3 && observation.c === 2) {
            return 3;
        }
        else if(observation.r === 3 && observation.c === 1) {
            return 3;
        }
        else if(observation.r === 3 && observation.c === 0) {
            return 2;
        }
        else if(observation.r === 4 && observation.c === 1) {
            return 3;
        }
        else if(observation.r === 4 && observation.c === 2) {
            return 3;
        }
        else if(observation.r === 4 && observation.c === 3) {
            return 3;
        }
        else if(observation.r === 4 && observation.c === 4) {
            return 3;
        }
        else {
            return 0;
        }
    }

    arrows() {
        let arrows = [];
        
    }
}

export { DeterministicAgent }