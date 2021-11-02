import { Environment } from "$lib/reinforcement_learning/common/Environment";

class GridEnvironment extends Environment {
    constructor(map, randomize=0) {
        let actionSpace = [0, 1, 2, 3];
        let observationSpace = [];
        for(let r = 0; r < map.rows; r++) {
            for(let c = 0; c < map.columns; c++) {
                observationSpace.push({r, c})
            }
        } 

        super(actionSpace, observationSpace);
        this.map = JSON.parse(JSON.stringify(map));
        this.initObservation = {... map.player};
        this.randomize = randomize;
    }

    getCells() {
        return JSON.parse(JSON.stringify(this.map.cells));
    }

    reset(){
        this.map.player = {... this.initObservation};
        return {... this.map.player}      
    }

    step(action){
        return this.model(action)
    }

    model(action) {
        let r = this.map.player.r;
        let c = this.map.player.c;

        //take random action
        if(Math.random() < this.randomize){
            let index = Math.floor(Math.random() * this.actionSpace.length);
            action = this.actionSpace[index]    
        }
        
        let player;
        //move but take care of grid boundaries
        switch(action) {
            case this.map.actions.north:
                player = {r: Math.max(0, r-1), c};
                break;
            case this.map.actions.east:
                player = {r, c: Math.min(this.map.columns-1, c+1)};
                break;
            case this.map.actions.south:
                player = {r: Math.min(this.map.rows-1, r+1), c};
                break;
            case this.map.actions.west:
                player = {r, c: Math.max(0, c-1)};
                break;
        }
        //move back if you landed on the block
        let cell;
        cell = this.findCell(player);
        if (cell.type === "block") {
            player = {r, c};
        }
        this.map.player = player;

        //calculate state, reward, done and return the payload
        cell = this.findCell(player);
        let reward =  cell.reward;
        let done = cell.type === "goal" ? true : false;
        let payload = {observation: {...player}, reward, done};
        return payload;
    }     
    
    findCell(address) {
        return this.map.cells.find(cell => {
            return cell.r === address.r && cell.c === address.c;
        })
    }
}

export {GridEnvironment}