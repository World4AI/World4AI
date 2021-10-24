<script>
    import Rectangle from '$lib/Rectangle.svelte'

    let data = [
        {
            x: 0.5,
            y: 0.5,
            width: 10,
            height: 50
        },
        {
            x: 20.5,
            y: 30.5,
            width: 10, 
            height: 20
        },
        {
            x: 40.5,
            y: 0.5,
            width: 10,
            height: 50
        },
        {
            x: 60.5,
            y: 40.5,
            width: 40,
            height: 10
        },
        {
            x: 60.5,
            y: 0.5,
            width: 10, 
            height: 40
        },
        {
            x:100.5, 
            y:10.5, 
            width:10, 
            height:40
        },
        {
            x:70.5,
            y:.5,
            width:40,
            height:10
        },
        {
            x:120.5,
            y:30.5,
            width:10,
            height:20
        },
        {
            x:160.5,
            y:30.5,
            width:10,
            height:20
        },
        {
            x:120.5,
            y:20.5,
            width:50,
            height:10
        },
        {
            x:120.5,
            y:.5,
            width:10,
            height:20
        },
        {
            x:160.5,
            y:10.5,
            width:10,
            height:10
        },
        {
            x:130.5,
            y:.5,
            width:40,
            height:10
        },
        {
            x:180.5,
            y:.5,
            width:10,
            height:50
        },
        {
            x:190.5,
            y:40.5,
            width:40,
            height:10
        },
        {
            x:240.5,
            y:40.5,
            width:40,
            height:10
        },
        {
            x:280.5,
            y:30.5,
            width:10,
            height:20
        },
        {
            x:240.5,
            y:20.5,
            width:10,
            height:20
        },
        {
            x:250.5,
            y:20.5,
            width:40,
            height:10
        },
        {
            x:280.5,
            y:.5,
            width:10,
            height:20
        },
        {
            x:340.5,
            y:30.5,
            width:10,
            height:20
        },
        {
            x:300.5,
            y:20.5,
            width:50,
            height:10
        },
        {
            x:340.5,
            y:.5,
            width:10,
            height:20
        },
        {
            x:300.5,
            y:.5,
            width:10,
            height:20
        },
        {
            x:360.5,
            y:30.5,
            width:10,
            height:20
        },
        {
            x:400.5,
            y:30.5,
            width:10,
            height:20
        },
        {
            x:360.5,
            y:20.5,
            width:50,
            height:10
        },
        {
            x:400.5,
            y:10.5,
            width:10,
            height:10
        },
        {
            x:360.5,
            y:.5,
            width:10,
            height:20
        },
        {
            x:370.5,
            y:.5,
            width:40,
            height:10
        },
        {
            x:420.5,
            y:40.5,
            width:50,
            height:10
        },
        {
            x:440.5,
            y:10.5,
            width:10,
            height:30
        },
        {
            x:420.5,
            y:.5,
            width:50,
            height:10
        }
    ]
</script>

<h1>Blocks</h1>
<h2>How is World4AI(W4AI) organized?</h2>
<p>World4AI is organized into building blocks. Each block builds knowledge that is fundamental to modern artificial intelligence. But you don't need to be proficient in any of the sections to start the next one, therefore you do not need to learn the blocks in a stricly linear manner. You can learn the material that you require as you go along. The sections will generally link to relevant posts from previous sections if the need arises.</p>

<svg  width="471" height="51" version="1.1" viewBox="0 0 471 51" xmlns="http://www.w3.org/2000/svg">
    <g fill="none" stroke="#000">
        {#each data as rect, index}
            <Rectangle delayFactor={index}  x={rect.x} y={rect.y} width={rect.width} height={rect.height}/>
        {/each}
    </g>
</svg>

<p>At the moment only a few sections have been started and none is considered finalized. More chapters will be published in the next weeks and months. But the overall process will most likely take many years.</p>

<style>
    svg {
        height: 100%;
        width: 100%;
    }

</style>