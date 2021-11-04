<script>
    import { onMount } from 'svelte';

    export let width = 750;
    export let height = 500;
    // TODO
    // Find a way to figure out global css colors
    export let color = '#dad9eb';

    let heightPaddle = 120;
    let widthPaddle = 20;
    let ballSize = 20;

    let canvas;

    let ballX;
    let ballY;
    let ballOffset;

    let paddleLeftX;
    let paddleRightX;

    let paddleLeftY;
    let paddleRightY;
    let paddleOffset;

    let ballVelX;
    let ballVelY;

    function reset() {
        ballOffset = ballSize / 2;
        ballX = width / 2 - ballOffset;
        ballY = height / 2 - ballOffset;

        paddleLeftX = 10;
        paddleRightX = width - widthPaddle - 10;

        paddleOffset = heightPaddle / 2
        paddleLeftY = height / 2 - paddleOffset;
        paddleRightY = height / 2 - paddleOffset;

        // reset ball velocity
        ballVelX = -1.5;
        ballVelY = -1.5;
        
    }

    function calculateNewPositions() {
        ballX += ballVelX;
        ballY += ballVelY;

        // reflect the ball from the walls
        if (ballY <= 0) {
            ballVelY *= -1;
        }
        if (ballY + ballSize >= height) {
            ballVelY *= -1;
        }

        //reflect the ball from the paddles
        // TODO calcualate end of game and make reflections exact for y coordinates of the paddle and the ball
        if (ballX <= paddleLeftX + widthPaddle) {
            ballVelX *= -1;
        }

        if (ballX + ballSize >= paddleRightX) {
            ballVelX *= -1;
        }

        let speed;
        //move left paddle
        speed = ballX < width / 2 ? 0.05 : 0.02;
        paddleLeftY = speed * (ballY - paddleLeftY - paddleOffset) + paddleLeftY;
        
        if (paddleLeftY + heightPaddle >= height)  {
            paddleLeftY = height - heightPaddle;
        } 

        if (paddleLeftY <= 0)  {
            paddleLeftY = 0;
        } 

        //move the right paddle
        speed = ballX < width / 0.05 ? 0.02 : 1;
        paddleRightY = speed * (ballY - paddleRightY - paddleOffset) + paddleRightY;
        if (paddleRightY + heightPaddle >= height)  {
            paddleRightY = height - heightPaddle;
        } 
        if (paddleRightY <= 0)  {
            paddleRightY = 0;
        } 

    }

    reset()
    onMount(() => {
        const ctx = canvas.getContext('2d');
        let frame = requestAnimationFrame(loop);

        function loop() {
            frame = requestAnimationFrame(loop)
            ctx.strokeStyle = color;

            ctx.clearRect(0, 0, width, height);
            calculateNewPositions();

            // left paddle
            ctx.strokeRect(paddleLeftX, paddleLeftY, widthPaddle, heightPaddle);
            // right paddle
            ctx.strokeRect(paddleRightX, paddleRightY, widthPaddle, heightPaddle);
            // ball
            ctx.strokeRect(ballX, ballY, ballSize, ballSize)

            
        }

        return () => {
            cancelAnimationFrame(frame);
        }
    })
</script>


<canvas bind:this={canvas} {width} {height}>
    The game of Pong
</canvas>

<style>

</style>