<script>
import { tweened } from 'svelte/motion';
import { draw, fade } from 'svelte/transition';

let run = 0;
let width = 500;
let height = 200;
let stateSize = 50;
let padding = 20;
let markerWidth = 6;
let markerHeight = 3;
let probFontSize = 12;
let lineOffset = 0.8;

let activeState = 0;
let nextState = 0;
//set the transition probabilities from one state to the next
export let transitions = [[0.2, 0.8], [0.5, 0.5]];

let mdp = () => {
  let probs = transitions[activeState];
  if(Math.random() < probs[0]){
    return 0;
  } else {
    return 1;
  }
};

//set the center of the boxes
let yCenter = height / 2;
let xCenters = []; 
xCenters.push(stateSize / 2 + padding);
xCenters.push(width-stateSize / 2 - padding)
  
//calculate the coordinates of the paths
//these are the paths from one state to the next
export let points = [
  [
    [
      {
        x: xCenters[0] + stateSize/2 * lineOffset,
        y: yCenter + stateSize/2
      },
      {
        x: xCenters[0] + stateSize/2 * lineOffset,
        y: yCenter + stateSize/2 + 30
      },
      {
        x: xCenters[0] - stateSize/2 * lineOffset,
        y: yCenter + stateSize/2 + 30
      },
      {
        x: xCenters[0] - stateSize/2 * lineOffset,
        y: yCenter + stateSize/2
      }
    ],
    [
      {
        x: xCenters[0] + stateSize/2, 
        y: yCenter - stateSize/2 * lineOffset 
      },
      { x: xCenters[1] - stateSize/2,
        y: yCenter - stateSize/2 * lineOffset 
      } 
    ]
  ],
  [
    [
      {
        x: xCenters[1] - stateSize/2, 
        y: yCenter + stateSize/2 * lineOffset 
      },
      { x: xCenters[0] + stateSize/2,
        y: yCenter + stateSize/2 * lineOffset 
      } 
    ],
    [
      {
        x: xCenters[1] - stateSize/2 * lineOffset,
        y: yCenter + stateSize/2
      },
      {
        x: xCenters[1] - stateSize/2 * lineOffset,
        y: yCenter + stateSize/2 + 30
      },
      {
        x: xCenters[1] + stateSize/2 * lineOffset,
        y: yCenter + stateSize/2 + 30
      },
      {
        x: xCenters[1] + stateSize/2 * lineOffset,
        y: yCenter + stateSize/2
      }
    ]
  ]
];
//coordinates of the probabilities text
let textCoordinates = [
  [
    {
      x: xCenters[0] + stateSize/2 * lineOffset - 20,
      y: yCenter + stateSize/2 + 20
    },
    {
      x: xCenters[0] + stateSize/2 + 30, 
      y: yCenter - stateSize/2 * lineOffset - 10
    }
  ],
  [
    {
      x: xCenters[1] - stateSize/2 * lineOffset - 25,
      y: yCenter + stateSize/2 - 15
    },
    {
      x: xCenters[1] - stateSize/2 + 25, 
      y: yCenter + stateSize/2 * lineOffset + 20 
    },
  ]
]

//create an svg path from the points
let coordinatesToLine = (coordinates) => {
 let path = `M${coordinates.map(coordinate => `${coordinate.x},${ coordinate.y}`).join('L')}` 
 return path;
}  

function resolveState(time) {
  return new Promise(resolve => {
    setTimeout(() => {
     resolve(mdp(activeState));  
    }, time);
  })
};


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
  nextState = await resolveState(1000);
  await wait(2000);
  process.unshift(nextState); 
  if(process.length > 20) {
      process.pop()
  }
  process = [... process];
  run+=1;
}
runAnimations();
$: run && runAnimations();

</script>

<svg viewBox='0 0 {width} {height}'> 
    <defs>
    <marker id="arrowhead" markerwidth={markerWidth} markerheight={markerHeight} refX="0" refY={markerHeight/2} orient="auto" fill="var(--text-color)">
        <polygon points="0 0, {markerWidth} {markerHeight/2}, 0 {markerHeight}" />
    </marker>
    </defs>
    <!--draw states -->
    {#each transitions as transition, i}
      {#if i == activeState}
        <rect transition:fade stroke="black" fill="var(--main-color-1)" opacity="0.2" x={xCenters[i] - stateSize / 2 - 5} y={yCenter - stateSize / 2 - 5} width={stateSize + 10} height={stateSize + 10} />
      {/if}
      <rect stroke="black" fill="var(--main-color-1)" x={xCenters[i]-stateSize/2} y={yCenter-stateSize/2} width={stateSize} height={stateSize} />
      <text stroke="none" fill="var(--background-color)" x={xCenters[i]} y={yCenter} dominant-baseline="middle" text-anchor="middle">
      {i}
      </text>
    {/each}
    <!-- draw connections -->
    {#each points as state, i}
      {#each state as coordinates, k}
      <path d={coordinatesToLine(coordinates)} 
        marker-end=url(#arrowhead)
        stroke="var(--text-color)"
        fill="transparent"
        />
      <text stroke="none" fill={i==activeState && k==nextState ? "var(--main-color-1)" : "var(--text-color)"} x={textCoordinates[i][k].x} y={textCoordinates[i][k].y} dominant-baseline="middle" text-anchor="middle">
      {transitions[i][k]}
      </text>
        <!-- draw active connection -->
        {#if i==activeState && k == nextState} 
         <path in:draw="{{duration: 1000}}" d={coordinatesToLine(coordinates)} 
           marker-end=url(#arrowhead)
           stroke-width=2
           stroke="var(--main-color-1)"
           fill="transparent"
           />
                  
        {/if}
      {/each}
    {/each}
    {#each process as state, i}
      <circle cx={width - i*35 - 20} cy={20} r={15} stroke="black" fill="var(--main-color-1)" />
      <text dominant-baseline="middle" text-anchor="middle" x={width - i*35 - 20} y={20} fill="black">{state}</text>
    {/each}
</svg>

