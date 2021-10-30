class Environment {
    constructor(actionSpace = null, observationSpace = null) {
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
    }

    reset(){}

    step(){}

    model(){}
}

export {Environment}