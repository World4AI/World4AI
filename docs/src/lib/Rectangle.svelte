<script>
    import { tweened } from 'svelte/motion';
	import { sineInOut } from 'svelte/easing';

    export let delayFactor;
    export let x;
    export let y;
    export let height;
    export let width;

    const progress = tweened(-100, {
		duration: 800,
        delay: delayFactor * 100,
		easing: sineInOut
	});
    

    progress.set(y)
    
</script>

<rect fill="var(--main-color-2)" x={x} y={$progress} width={width} height={height}/>