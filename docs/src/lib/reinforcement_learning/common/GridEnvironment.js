import { Environment } from "$lib/reinforcement_learning/common/Environment";

class GridEnvironment extends Environment {
    constructor(rows, columns, obstacles, goal) {

        let actionSpace = [0, 1, 2, 3]
        let observationSpace = []
        for(let r = 0; r < rows; r++) {
            for(let c = 0; c < columns; c++) {
                observationSpace.push({r, c})
            }
        }
        super(actionSpace, observationSpace);

        this.rows = rows;
        this.columns = columns;
        this.goal = goal;
        this.obstacles = obstacles;
        this.north = 0;
        this.east = 1;
        this.south = 2;
        this.west = 3;

        // initial state
        this.initObservation = {r: 0, c: 0};
        this.reset();
    }

    reset(){
        this.observation = {... this.initObservation};
        return {... this.observation}
    }

    step(action){
        return this.model(action)
    }

    model(action){
        let observation = this.calculateMovement(action);
        let isBlocked = this.isBlocked(observation);
        let isDone = this.isDone(observation);
        let reward = this.calculateReward(observation);
        
        if (isBlocked) {
            observation = {... this.observation};
        }
        this.observation = {... observation};

        if (isDone) {
            this.reset();
        }

        return {observation, reward, isDone}
    }

    calculateMovement(action) {
        let r = this.observation.r;
        let c = this.observation.c;
        let newObservation;
        switch(action) {
            case this.north:
                newObservation = {r: Math.max(0, r-1), c};
                break;
            case this.east:
                newObservation = {r, c: Math.min(this.columns-1, c+1)};
                break;
            case this.south:
                newObservation = {r: Math.min(this.rows-1, r+1), c};
                break;
            case this.west:
                newObservation = {r, c: Math.max(0, c-1)};
                break;
        }
        return newObservation;
    }

    isBlocked(observation) {
        return this.obstacles.some((obstacle) => { 
            return obstacle.r === observation.r && obstacle.c === observation.c;
        });
    }

    isDone(observation) {
        if(observation.r === this.goal.r && observation.c === this.goal.c) {
            return true;
        }
        return false;
    }

    calculateReward(observation) {
        if(observation.r === this.goal.r && observation.c === this.goal.c) {
            return 1;
        }
        return -1;
    }
}

export {GridEnvironment}