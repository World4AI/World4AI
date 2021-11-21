<script>
import State from './State.svelte';
import Reward from './Reward.svelte';
import Action from './Action.svelte'; 
import Connection from './Connection.svelte';

export let showRewards = false;
export let showActions = false;
let timeStep = 0; 
let width = 500;
let height = 300;
let stateSize = 50;
let paddingLeftRight = 70;
let paddingTopBottom = 0;
let markerWidth = 6;
let markerHeight = 3;
let probFontSize = 12;
let lineOffset = 0.8;
let actionOffset = 80;

let activeState = 0;
let nextState = 0;
let nextAction = 0;
let nextReward = null;
//set the transition probabilities from one state to the next
export let transitionsWithoutActions = [[[0.2, 0.8]], [[0.5, 0.5]]];
export let transitionsWithActions = [[[0.8, 0.2], [0.1, 0.9]], [[0.7, 0.3], [0.3, 0.7]]]
export let actions = [[0.3, 0.7], [0.4, 0.6]]

let transitions;
if (showActions){
  transitions = transitionsWithActions;
} else {
  transitions = transitionsWithoutActions;
}

let mdp = () => {
  let probs = transitions[activeState][nextAction];
  if(Math.random() < probs[0]){
    return 0;
  } else {
    return 1;
  }
};

let takeAction = () => {
  let probs = actions[activeState];
  if(Math.random() < probs[0]){
    return 0;
  } else {
    return 1;
  }
};

//set the center of the boxes
let yCenter = height / 2 + paddingTopBottom;
let stateCenters = []; 
stateCenters.push({x: stateSize / 2 + paddingLeftRight, y: yCenter});
stateCenters.push({x: width-stateSize / 2 - paddingLeftRight, y: yCenter});

//set the center of the circles
let actionCenters = [[], []];
//actions for the left (0) state
actionCenters[0].push({x: stateCenters[0].x + actionOffset, y: stateCenters[0].y - actionOffset})
actionCenters[0].push({x: stateCenters[0].x + actionOffset, y: stateCenters[0].y + actionOffset})
actionCenters[1].push({x: stateCenters[1].x - actionOffset, y: stateCenters[1].y - actionOffset})
actionCenters[1].push({x: stateCenters[1].x - actionOffset, y: stateCenters[1].y + actionOffset})
  
//calculate the coordinates of the paths
//these are the paths from one state to the next
export let noActionsPoints = [
  [
    [
      [
        {
          x: stateCenters[0].x + stateSize/2 * lineOffset,
          y: stateCenters[0].y + stateSize/2
        },
        {
          x: stateCenters[0].x + stateSize/2 * lineOffset,
          y: stateCenters[0].y + stateSize/2 + 30
        },
        {
          x: stateCenters[0].x - stateSize/2 * lineOffset,
          y: stateCenters[0].y + stateSize/2 + 30
        },
        {
          x: stateCenters[0].x - stateSize/2 * lineOffset,
          y: stateCenters[0].y + stateSize/2
        }
      ],
      [
        {
          x: stateCenters[0].x + stateSize/2, 
          y: stateCenters[0].y - stateSize/2 * lineOffset 
        },
        { x: stateCenters[1].x - stateSize/2,
          y: stateCenters[1].y - stateSize/2 * lineOffset 
        } 
      ]
    ],
  ],
  [
    [
      [
        {
          x: stateCenters[1].x - stateSize/2, 
          y: stateCenters[1].y + stateSize/2 * lineOffset 
        },
        { x: stateCenters[0].x + stateSize/2,
          y: stateCenters[0].y + stateSize/2 * lineOffset 
        } 
      ],
      [
        {
          x: stateCenters[1].x - stateSize/2 * lineOffset,
          y: stateCenters[1].y + stateSize/2
        },
        {
          x: stateCenters[1].x - stateSize/2 * lineOffset,
          y: stateCenters[1].y + stateSize/2 + 30
        },
        {
          x: stateCenters[1].x + stateSize/2 * lineOffset,
          y: stateCenters[1].y + stateSize/2 + 30
        },
        {
          x: stateCenters[1].x + stateSize/2 * lineOffset,
          y: stateCenters[1].y + stateSize/2
        }
      ]
    ]
  ]
];

export let actionsPoints = [
  [
    [
      [
        { 
          x: actionCenters[0][0].x,
          y: actionCenters[0][0].y
        },
        {
          x: stateCenters[0].x,
          y: actionCenters[0][0].y 
        },
        {
          x: stateCenters[0].x,
          y: stateCenters[0].y - stateSize / 2
        },
      ],
      [
        {
          x: actionCenters[0][0].x,
          y: actionCenters[0][0].y
        },
        { x: stateCenters[1].x - stateSize / 2,
          y: stateCenters[1].y - 10
        } 
      ]
    ],
    [
      [
        { 
          x: actionCenters[0][1].x,
          y: actionCenters[0][1].y
        },
        {
          x: stateCenters[0].x,
          y: actionCenters[0][1].y 
        },
        {
          x: stateCenters[0].x,
          y: stateCenters[0].y + stateSize / 2
        },
      ],
      [
        {
          x: actionCenters[0][1].x,
          y: actionCenters[0][1].y
        },
        { x: stateCenters[1].x - stateSize / 2,
          y: stateCenters[1].y + 10
        } 
      ]
    ]
  ],
  [
    [
      [
        {
          x: actionCenters[1][0].x,
          y: actionCenters[1][0].y
        },
        { x: stateCenters[0].x + stateSize / 2,
          y: stateCenters[0].y - 10
        } 
      ],
      [
        { 
          x: actionCenters[1][0].x,
          y: actionCenters[1][0].y
        },
        {
          x: stateCenters[1].x,
          y: actionCenters[1][0].y 
        },
        {
          x: stateCenters[1].x,
          y: stateCenters[1].y - stateSize / 2
        },
      ]
    ],
    [
      [
        {
          x: actionCenters[1][1].x,
          y: actionCenters[1][1].y
        },
        { x: stateCenters[0].x + stateSize / 2,
          y: stateCenters[0].y + 10
        } 
      ],
      [
        { 
          x: actionCenters[1][1].x,
          y: actionCenters[1][1].y
        },
        {
          x: stateCenters[1].x,
          y: actionCenters[1][1].y 
        },
        {
          x: stateCenters[1].x,
          y: stateCenters[1].y + stateSize / 2
        },
      ]
    ]
  ]
];

let points;
if(showActions) {
  points = actionsPoints;
} else {
  points = noActionsPoints;
}
  
//coordinates of the probabilities text
let noActionsTextCoordinates = [
  [
    [
      {
        x: stateCenters[0].x + stateSize/2 * lineOffset - 20,
        y: stateCenters[1].y + stateSize/2 + 20
      },
      {
        x: stateCenters[0].x + stateSize/2 + 30, 
        y: stateCenters[0].y - stateSize/2 * lineOffset - 10
      }
    ],
  ],
  [
    [
      {
        x: stateCenters[1].x - stateSize/2 * lineOffset - 25,
        y: stateCenters[1].y + stateSize/2 - 15
      },
      {
        x: stateCenters[1].x - stateSize/2 + 25, 
        y: stateCenters[1].y + stateSize/2 * lineOffset + 20 
      },
    ]
  ]
]

let actionsTextCoordinates = [
  [
    [
      {
        x: actionCenters[0][0].x - 30,
        y: actionCenters[0][0].y - 10
      },
      {
        x: actionCenters[0][0].x + 35,
        y: actionCenters[0][0].y 
      }
    ],
    [
      {
        x: actionCenters[0][1].x - 30,
        y: actionCenters[0][1].y + 15
      },
      {
        x: actionCenters[0][1].x + 40,
        y: actionCenters[0][1].y 
      }
    ],
  ],
  [
    [
      {
        x: actionCenters[1][0].x - 40,
        y: actionCenters[1][0].y + 2 
      },
      {
        x: actionCenters[1][0].x + 35,
        y: actionCenters[1][0].y - 10 
      },
    ],
    [
      {
        x: actionCenters[1][1].x - 30,
        y: actionCenters[1][1].y + 5
      },
      {
        x: actionCenters[1][1].x + 40,
        y: actionCenters[1][1].y - 10 
      },
    ],
  ]
]

let textCoordinates;
if(showActions) {
  textCoordinates = actionsTextCoordinates;
} else {
  textCoordinates = noActionsTextCoordinates;
}

let actionCoordinates = [
  [
    {
      x: stateCenters[0].x + 30,  
      y: stateCenters[0].y - 50
    },
    {
      x: stateCenters[0].x + 30,  
      y: stateCenters[0].y + 50
    }
  ],
  [
    {
      x: stateCenters[1].x - 30,  
      y: stateCenters[1].y - 50
    },
    {
      x: stateCenters[1].x - 30,  
      y: stateCenters[1].y + 50
    }
  ]
]

//parameters for rewards
let rewards = [
  {
    x : 20,
    y : height/2,
    rotate: 0, 
    rewardValue: 5
  },
  {
    x: width - 20,
    y: height / 2,
    rotate: 180,
    rewardValue: -1
  }
]


//create an svg path from the points
let coordinatesToLine = (coordinates) => {
 let path = `M${coordinates.map(coordinate => `${coordinate.x},${ coordinate.y}`).join('L')}` 
 return path;
}  

function resolveState(time) {
  return new Promise(resolve => {
    setTimeout(() => {
     resolve(mdp());  
    }, time);
  })
};

function resolveAction(time){
  return new Promise(resolve => {
    setTimeout(() => {
     resolve(takeAction());  
    }, time);
  })
};

function resolveReward(time){
  return new Promise(resolve => {
    setTimeout(() => {
     resolve(rewards[nextState].rewardValue);  
    }, time);
  })
}

function wait(time) {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve('waited');
    }, time);
  })
};

let process = [];
async function runAnimations() {
  activeState = nextState;
  nextState = null;
  nextReward = null;
  if(showActions){
    nextAction = null;
    nextAction = await resolveAction(1000);
    process.unshift({type: 'action', value: nextAction}); 
    await wait(2000);
    process = [... process];
  }
  nextState = await resolveState(1000);
  process.unshift({type: 'state', value: nextState}); 
  process = [... process];
  if(showRewards){
    nextReward = await resolveReward(1000);
    process.unshift({type: 'reward', value: nextReward});
    await wait(2000);
    process = [... process];
  }
  await wait(2000);
  
  while (process.length > 20){
      process.pop()
  }
  timeStep+=1;
}
runAnimations();
$: timeStep && runAnimations();

</script>

<svg viewBox='0 0 {width} {height}'> 
    <defs>
      <marker id="arrowhead" markerwidth={markerWidth} markerheight={markerHeight} refX="5" refY={markerHeight/2} orient="auto" fill="var(--text-color)">
          <polygon points="0 0, {markerWidth} {markerHeight/2}, 0 {markerHeight}" />
      </marker>
    </defs>

    <!-- draw connections -->
    {#each points as state, i}
      {#each state as action, a}
        {#each action as coordinates, k}
        <Connection  d={coordinatesToLine(coordinates)} 
                     xText={textCoordinates[i][a][k].x} 
                     yText={textCoordinates[i][a][k].y} 
                     active={i==activeState && nextAction==a && k==nextState}
                     probability={transitions[i][a][k]}
                     />
        {/each}
      {/each}
    {/each}

    <!-- draw actions -->
    {#if showActions}
      {#each actionCenters as state, i}
        {#each state as action, a}
          <Action cx={action.x} 
                  cy={action.y} 
                  fromX={stateCenters[i].x} 
                  fromY={stateCenters[i].y} 
                  actionValue={a} 
                  active={i==activeState && a==nextAction}
                  xText={actionCoordinates[i][a].x}
                  yText={actionCoordinates[i][a].y}
                  probability={actions[i][a]}
                  />
        {/each}
      {/each} 
    {/if}

    <!--draw states -->
    {#each transitions as transition, i}
      <State active={nextState==i} x={stateCenters[i].x-stateSize/2} y={stateCenters[i].y-stateSize/2} size={stateSize} stateNumber={i} />
    {/each}

    <!-- draw rewards -->
    {#if showRewards}
      {#each rewards as reward, i}
        <Reward cx={reward.x} cy={reward.y} degrees={reward.rotate} active={i==nextState && nextReward ? true : false} rewardValue={reward.rewardValue} />
      {/each}
    {/if}
  
    <!-- draw random variables from stochastic process -->
    {#each process as part, i}
      <circle cx={width - i*35 - 20} cy={20} r={15} stroke="black" fill={part.type == 'state' ?  'var(--main-color-1)' : part.type == 'reward' ? 'var(--text-color)' : 'var(--main-color-2)'} />
      <text dominant-baseline="middle" text-anchor="middle" x={width - i*35 - 20} y={20} fill="black">{part.value}</text>
    {/each}
</svg>

