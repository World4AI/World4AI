class Environment {
    constructor(actionSpace = null, observationSpace = null) {
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
    }

    getActionSpace() {
        return [... this.actionSpace];
    }

    getObservationSpace() {
        return [... this.observationSpace];
    }

    reset(){}

    step(){}

    model(){}
}

export {Environment}