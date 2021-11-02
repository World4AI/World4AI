<script>
    import { onMount } from 'svelte';

    export let agent;
    export let env;
    export let speed = 500;
    export let showArrows = false;
    export let isColoredReward = false;
    export let showObservation = false;
    export let showAction = false;
    export let showReward = false;
    export let showAllRewards = false;
    export let arrows = [];

    // svg parameters
    export let width = 500;
    export let height = 500;
    export let strokeWidth = 1;

    // if you need more room to display the information
    // like the current state or received reward
    let translateGrid = 0;

    // calcualte cell size
    let colSize = 100;
    let rowSize = 100;

    //agent environment interaction
    let observation = env.reset();
    let player = {... observation};
    let action = null;
    let reward = null;
    let cells = env.getCells();
    let payload = {};
    onMount(() => {
        if(showObservation || showAction || showReward) {
            translateGrid = 150;
        }
        const interval = setInterval(() => {
            if (payload.done) {
                    observation = env.reset();
                    payload.done = false;
                    player = {... observation};
                    action = null;
                    reward = null;
                }
            else {
                action = agent.act(observation);
                arrows = [{r: observation.r, c: observation.c, d: actionToDegreeMapping[action]}]
                payload = env.step(action);
                observation = payload.observation;
                reward = payload.reward;
                player = {... observation};
            }
        }, speed);

        return () => clearInterval(interval);
    })

    // map from action to the degrees of the arrow
    let actionToDegreeMapping = {
        0: 270,
        1: 0,
        2: 90,
        3: 180
    }

    //goal
    let goalPadding = 40;
    // obstacles
    let obstaclePadding = 10;
</script> 

<svg {width} height={height+translateGrid} version="1.1" viewBox="0 0 {width} {height+translateGrid}" xmlns="http://www.w3.org/2000/svg">
    {#if showObservation}
        <text x="0" y="25" font-size="30px" font-family="sans-serif" fill="var(--text-color)" stroke="none">State: ({observation.r}, {observation.c})</text>
    {/if}
    {#if showAction}
        <text x="0" y="70" font-size="30px" font-family="sans-serif" fill="var(--text-color)" stroke="none">Action: ({action})</text>
    {/if}
    {#if showReward}
        <text x="0" y="115" font-size="30px" font-family="sans-serif" fill="var(--text-color)" stroke="none">Reward: ({reward})</text>
    {/if}

    
    <g id="grid" transform="translate(0, {translateGrid})">    
        <!-- Create the cells-->
        {#each cells as cell}
            <!-- cells -->
            <g id="cells" fill="none" stroke="var(--text-color)" stroke-width={strokeWidth}>
                <rect fill={isColoredReward && cell.reward > 0 ? 'var(--main-color-2)' 
                : isColoredReward && cell.reward < 0 ? 'var(--main-color-1' : 'none'} 
                    x={cell.c * colSize} 
                    y={cell.r * rowSize} 
                    width={colSize} 
                    height={rowSize}/>
            </g>

            {#if cell.type !== "block"  && showAllRewards}
                <text 
                    x={5+cell.c * colSize} 
                    y={20+ + cell.r * rowSize} 
                    font-size="20px" 
                    font-family="sans-serif" 
                    fill="var(--text-color)" 
                    stroke="none">
                    {cell.reward}
                </text>
            {/if}

            <!-- blocks -->
            <g fill="var(--text-color)" stroke="black" stroke-width="3">
                {#if cell.type === "block"}
                    <rect 
                        x={cell.c * colSize + obstaclePadding} 
                        y={cell.r * rowSize + obstaclePadding} 
                        width={colSize - obstaclePadding * 2} 
                        height={rowSize - obstaclePadding * 2}>
                    </rect>
                {/if}
            </g>

            <!-- goal -->
            <g fill="var(--text-color)" stroke="black" stroke-width="2">
                {#if cell.type === "goal"}
                    <polygon points={`${cell.c * colSize + colSize/2},${cell.r * rowSize + goalPadding} \
                    ${cell.c * colSize + colSize - goalPadding},${cell.r * rowSize + rowSize - goalPadding} \
                    ${cell.c * colSize + goalPadding},${cell.r * rowSize + rowSize - goalPadding}`}/>
                {/if}
            </g>
        {/each}

        <!-- player -->
        <g>
            <circle cx={player.c * colSize + colSize / 2} 
                    cy={player.r * rowSize + rowSize / 2} 
                    r={colSize * 0.25} 
                    fill="var(--text-color)" 
                    opacity="0.8" 
                    stroke="black" 
                    stroke-width="3"
            />
        </g>
    

        {#if showArrows}
            <defs>
                <marker id="arrowhead" 
                markerWidth="10" 
                markerHeight="7" 
                refX="0" refY="3.5" 
                orient="auto" 
                fill="black"
                >
                <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
            </defs> -->
            <g>
            {#each arrows as arrow}
                <line 
                    x1={arrow.c * colSize + colSize / 2} 
                    y1={arrow.r * rowSize + rowSize / 2} 
                    x2={arrow.c * colSize + colSize - 35} 
                    y2={arrow.r * rowSize + rowSize / 2}
                    transform="rotate({arrow.d}, {arrow.c * colSize + colSize/2}, {arrow.r * rowSize + rowSize/2})" 
                    stroke="black"
                    stroke-width="1" 
                    marker-end="url(#arrowhead)" 
                />
            {/each}
            </g>
        {/if} 
    </g>
</svg>
