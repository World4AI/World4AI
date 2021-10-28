class Agent {
    constructor(observationSpace, actionSpace) {
        this.observationSpace = observationSpace;
        this.actionSpace = actionSpace;
    }

    act(state) {
        return Math.floor(Math.random() * this.actionSpace.length);
    }
}

export { Agent }