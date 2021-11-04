<script>
    import { onMount } from 'svelte';
    let offset = 0;
    export let texts = [
        {
            content: 'X',
            x: 0.5,
            y: 60
        },
        {
            content: 'Y',
            x: 475,
            y: 60
        }
    ]

    onMount(() => {
        let interval = setInterval(() => {
            offset += 20;
        }, 500);

        return () => clearInterval(interval);
    })

</script>

<svg width="500" height="100" version="1.1" viewBox="0 0 500 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
     <marker id="TriangleOutL" overflow="visible" orient="auto">
      <path transform="scale(.8)" d="m5.77 0-8.65 5v-10z" fill="context-stroke" fill-rule="evenodd" stroke="context-stroke" stroke-width="1pt"/>
     </marker>
    </defs>
    <g fill="var(--text-color)" stroke="var(--text-color)">
        <rect x="202.21" y="2.2133" width="95.573" height="95.573" fill="none" stroke-linejoin="round"/>
        {#each texts as text}
        <text stroke="none" x={text.x} y={text.y} font-family="sans-serif" font-size="25px" >
            {text.content}
        </text>
        {/each}
        <g stroke-dashoffset={offset} fill="none">
        <path  d="m29.328 50h165.54" marker-end="url(#TriangleOutL)" stroke-dasharray="8,2,1,2"/>
        <path d="m303.25 50h162.86" marker-end="url(#TriangleOutL)" stroke-dasharray="8, 2, 1, 2"/>
        <circle cx="250" cy="50" r="35.84" stroke-dasharray="8,2,1,2" stroke-linejoin="round"/>
        </g>
    </g>
</svg>