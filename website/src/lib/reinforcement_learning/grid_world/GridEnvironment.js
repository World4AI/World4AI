import { Environment } from "$lib/reinforcement_learning/grid_world/Environment";
import { writable } from 'svelte/store';

class GridEnvironment extends Environment {
    constructor(map, random=false) {
        let actionSpace = Object.values(map.actions);
        let observationSpace = [];
        for(let r = 0; r < map.rows; r++) {
            for(let c = 0; c < map.columns; c++) {
                observationSpace.push({r, c})
            }
        } 

        super(actionSpace, observationSpace);
        this.random = random;
        this.randomness = 0.5;
        this.map = JSON.parse(JSON.stringify(map));
        this.initObservation = {... map.player};

        this.cellsStore = writable(this.getCells());
    }

    getCells() {
        return JSON.parse(JSON.stringify(this.map.cells));
    }

    getPlayer() {
        return {... this.map.player}      
    }

    reset(){
        this.map.player = {... this.initObservation};
        //reset store
        this.cellsStore.set(this.getCells());
        return {... this.map.player}      
    }

    step(action){
        //if environment is stochastic
        if(this.random){
          if(Math.random() < this.randomness){
              let index = Math.floor(Math.random() * this.actionSpace.length);
              action = this.actionSpace[index]    
          }
        }
        return this.model(action)
    }
 
    randomAction(){
      let index = Math.floor(Math.random() * this.actionSpace.length);
      action = this.actionSpace[index]    
      return action
    }

    getModel(){
      let P = {}; 
      for (let row = 0; row < this.map.rows; row++) {
        P[row] = {};
        for (let column = 0; column < this.map.columns; column++){
          P[row][column] = {};  
          for (let actionIndex = 0; actionIndex < this.actionSpace.length; actionIndex++) {
            P[row][column][actionIndex] = [];
            for (let direction = 0; direction < this.actionSpace.length; direction++) { 
              let probability;
              if (direction === actionIndex) {
                probability = this.randomness + (1- this.randomness) / this.actionSpace.length;
              } else {
                probability = (1 - this.randomness) / this.actionSpace.length;
              }
              let cell;
              let observation = {r: row, c: column};
              let reward = 0;
              cell = this.findCell(observation);
              if (cell.type != "goal" && cell.type != "block"){
                observation = this.modelBoundaries(direction, {r: row, c: column});
                cell = this.findCell(observation);
                reward = cell.reward;
              }
              let done = cell.type === "goal" ? true : false;
              P[row][column][actionIndex].push({probability, observation, reward, done}); 
            }
          }
        }
      }
      return P;
    }

    modelBoundaries(action, observation){
        let player;
        //move but take care of grid boundaries
        switch(action) {
            case this.map.actions.north:
                player = {r: Math.max(0, observation.r-1), c: observation.c};
                break;
            case this.map.actions.east:
                player = {r: observation.r, c: Math.min(this.map.columns-1, observation.c+1)};
                break;
            case this.map.actions.south:
                player = {r: Math.min(this.map.rows-1, observation.r+1), c: observation.c};
                break;
            case this.map.actions.west:
                player = {r: observation.r, c: Math.max(0, observation.c-1)};
                break;
        }
        //move back if you landed on the block
        let cell;
        cell = this.findCell(player);
        if (cell.type === "block") {
            player = {r: observation.r, c: observation.c};
        }
        return player;
    }

    model(action) {
        let player = this.modelBoundaries(action, this.map.player);
        this.map.player = player;

        //calculate state, reward, done and return the payload
        let cell = this.findCell(player);
        let reward =  cell.reward;
        let done = cell.type === "goal" ? true : false;
        let payload = {observation: {...player}, reward, done};

        this.cellsStore.set(this.getCells());
        return payload;
    }     
    
    findCell(address) {
        return this.map.cells.find(cell => {
            return cell.r === address.r && cell.c === address.c;
        })
    }
}

export {GridEnvironment}
