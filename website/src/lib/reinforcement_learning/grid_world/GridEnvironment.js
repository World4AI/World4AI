import { Environment } from "$lib/reinforcement_learning/grid_world/Environment";
import { writable } from 'svelte/store';

class GridEnvironment extends Environment {
    constructor(map) {
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

        this.cells = writable(this.getCells());
        this.player = writable(this.getPlayer());
    }

    getCells() {
        return JSON.parse(JSON.stringify(this.map.cells));
    }

    getPlayer() {
        return {... this.map.player}      
    }

    reset(){
        this.map.player = {... this.initObservation};
        return {... this.map.player}      
    }

    step(action){
        return this.model(action)
    }
 
    randomAction(){
      let index = Math.floor(Math.random() * this.actionSpace.length);
      action = this.actionSpace[index]    
      return action
    }

    modelBoundaties(action){
        let player;
        //move but take care of grid boundaries
        switch(action) {
            case this.map.actions.north:
                player = {r: Math.max(0, this.map.player.r-1), c: this.map.player.c};
                break;
            case this.map.actions.east:
                player = {r: this.map.player.r, c: Math.min(this.map.columns-1, this.map.player.c+1)};
                break;
            case this.map.actions.south:
                player = {r: Math.min(this.map.rows-1, this.map.player.r+1), c: this.map.player.c};
                break;
            case this.map.actions.west:
                player = {r: this.map.player.r, c: Math.max(0, this.map.player.c-1)};
                break;
        }
        return player;
    }

    model(action) {
        let player = this.modelBoundaties(action);

        //move back if you landed on the block
        let cell;
        cell = this.findCell(player);
        if (cell.type === "block") {
            player = {r: this.map.player.r, c: this.map.player.c};
        }
        this.map.player = player;

        //calculate state, reward, done and return the payload
        cell = this.findCell(player);
        let reward =  cell.reward;
        let done = cell.type === "goal" ? true : false;
        let payload = {observation: {...player}, reward, done};

        this.cells.set(this.getCells());
        this.player.set(this.getPlayer());
        return payload;
    }     
    
    findCell(address) {
        return this.map.cells.find(cell => {
            return cell.r === address.r && cell.c === address.c;
        })
    }
}

export {GridEnvironment}
