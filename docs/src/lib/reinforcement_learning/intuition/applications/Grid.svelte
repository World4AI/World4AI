<script>
    import { onMount } from 'svelte';
    import {Environment} from './environment.js'
    import {Agent} from './agent.js';

    // grid parameters
    export let columns = 5;
    export let rows = 5;
    export let player = {
        r: 0,
        c: 0
    }
    export let obstacles = [
        {
            r: 2,
            c: 0 
        },
        {
            r: 2,
            c: 1 
        },
        {
            r: 2,
            c: 2 
        },
    ]
    export let goal = {
        r: 4,
        c: 0
    };

    export let arrows = []

/*     export let arrows = [
        {
            r: 0,
            c: 0,
            d: 0
        },
        {
            r: 1,
            c: 1,
            d: 90
        }
    ] */

    const env = new Environment(rows, columns, player, obstacles, goal);
    const agent = new Agent(env.observationSpace, env.actionSpace);

    let observation = env.reset()
    onMount(() => {
        const interval = setInterval(() => {
            let action = agent.act(observation);
            observation = env.step(action);
            let coordinates = env.statesToCoordinates(observation);

            translateX = coordinates.col * colSize;
            translateY = coordinates.row * rowSize;
        }, 500);

        return () => clearInterval(interval);
    })

    let translateX = 0;
    let translateY = 0;

    // svg parameters
    export let width = 500;
    export let height = 500;
    export let strokeWidth = 1;

    // calcualte cell size
    let colSize = width / columns;
    let rowSize = height / rows;

    //player parameters
    let r = colSize * 0.5 * 0.5;
    let cx = player.c * colSize + colSize / 2;
    let cy = player.r * rowSize + rowSize / 2;

    //goal parameters
    let goalPadding = 40;

    // the triangle that is calculated based on the row and col location, while taking the padding into consideration
    let points = `${goal.c * colSize + colSize/2},${goal.r * rowSize + goalPadding} \
${goal.c * colSize + colSize - goalPadding},${goal.r * rowSize + rowSize - goalPadding} \
${goal.c * colSize + goalPadding},${goal.r * rowSize + rowSize - goalPadding}`;

    // obstacles
    let obstaclePadding = 10;
</script> 

<svg {width} {height} version="1.1" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    

    <!-- Create the grid-->
    <g id="grid" fill="none" stroke="var(--text-color)" stroke-width={strokeWidth}>
        {#each Array(columns+1) as _, col}
            <path d="m{col * colSize} 0 v{height}" />
        {/each}
        {#each Array(rows+1) as _, row}
            <path d="m0 {row * rowSize} h{width}" />
        {/each}
    </g>
    <g class="obstacles" fill="none" stroke="var(--text-color)">
        {#each obstacles as obstacle }
            <rect 
                x={obstacle.c * colSize + obstaclePadding} 
                y={obstacle.r * rowSize + obstaclePadding} 
                width={colSize - obstaclePadding * 2} 
                height={rowSize - obstaclePadding * 2}>
            </rect>
        {/each}
    </g>
    <g id="player" stroke="var(--text-color)" transform="translate({translateX} {translateY})">
        <circle id="player" {cx} {cy} {r} fill="none" stroke-width="1"/>
    </g>
    <g id="goal" fill="none" stroke="var(--text-color)">
        <polygon {points}/>
    </g>

    <defs>
        <marker id="arrowhead" 
        markerWidth="10" 
        markerHeight="7" 
        refX="0" refY="3.5" 
        orient="auto" 
        fill="var(--text-color)"
        >
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>
    </defs>
    {#each arrows as arrow}
        <!--push the arrows in the middle-->
        <line x1={arrow.c * colSize + colSize/2} 
        y1={arrow.r * rowSize + rowSize/2} 
        x2={arrow.c * colSize + colSize - 20} 
        y2={arrow.r * rowSize + rowSize/2}
        transform="rotate({arrow.d}, {arrow.c * colSize + colSize/2}, {arrow.r * rowSize + rowSize/2})" 
        stroke="var(--text-color)" 
        stroke-width="1" 
        marker-end="url(#arrowhead)" />
    {/each}
    
</svg>
