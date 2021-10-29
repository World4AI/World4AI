class Environment {
    constructor(rows, columns, player, obstacles, goal) {
        this.columns = columns;
        this.rows = rows;
        this.obstacles = [];

        for (let obstacle of obstacles) {
            let state = this.coordinatesToStates(obstacle.c, obstacle.r)
            this.obstacles.push(state)
        }

        this.goal = this.coordinatesToStates(goal.c, goal.r)
        this.initialState = this.coordinatesToStates(player.c, player.r);

        this.top = 3;
        this.right = 0;
        this.down = 1;
        this.left = 2;
        this.actionSpace = [0, 1, 2, 3];
        this.observationSpace = [];

        for (let i=0; i < columns * rows; i++){
            this.observationSpace.push(i);
        }

        this.createModel()
    }

    coordinatesToStates(c, r){
        return r*this.columns + c
    }

    statesToCoordinates(s) {
        let r = Math.floor(s / this.columns);
        let c = s % this.columns;

        return {r, c}
    }

    createModel() {
        this.model = {}
        for (let c = 0; c < this.columns; c++) {
            for(let r = 0; r < this.rows; r++) {
                let top = this.coordinatesToStates(c, Math.max(0, r-1));
                if (this.obstacles.includes(top)){
                    top = this.coordinatesToStates(c, r)
                }

                let right = this.coordinatesToStates(Math.min(this.columns-1, c+1), r);
                if (this.obstacles.includes(right)){
                    right = this.coordinatesToStates(c, r)
                }

                let down = this.coordinatesToStates(c, Math.min(this.rows-1, r+1));
                if (this.obstacles.includes(down)){
                    down = this.coordinatesToStates(c, r)
                }

                let left = this.coordinatesToStates(Math.max(0, c-1), r);
                if (this.obstacles.includes(left)){
                    left = this.coordinatesToStates(c, r)
                }

                this.model[this.coordinatesToStates(c, r)] = {
                    [this.top]: top,
                    [this.right]: right,
                    [this.down]: down,
                    [this.left]: left,
                }
            }
        }
    }

    observationSpace() {
        return [... this.stateSpace];
    }

    actionSpace() {
        return [... this.actionSpace];
    }

    step(action) {
        if (this.state == this.goal)
        {
            return this.reset()
        }
        this.state =  this.model[this.state][action];
        return this.state;
    }

    reset() {
        this.state = this.initialState;
        return this.state;
    }
}

export { Environment }