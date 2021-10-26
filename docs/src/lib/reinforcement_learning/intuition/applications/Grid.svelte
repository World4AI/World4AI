<script>
    import { onMount } from 'svelte';

    let state = 0

    const policy = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 9,
        9: 14,
        14: 19,
        19: 24,
        24: 23,
        23: 22,
        22: 21,
        21: 20,
        20: 0
    }

    const translations = {
        0: {
            translateX: 0, 
            translateY: 0
        },
        1: {
            translateX: 100,
            translateY: 0
        },
        2: {
            translateX: 200,
            translateY: 0
        },
        3: {
            translateX: 300,
            translateY: 0
        },
        4: {
            translateX: 400,
            translateY: 0
        },
        9: {
            translateX: 400,
            translateY: -100
        },
        14: {
            translateX: 400,
            translateY: -200
        },
        19: {
            translateX: 400,
            translateY: -300
        },
        24: {
            translateX: 400,
            translateY: -400
        },
        23: {
            translateX: 300,
            translateY: -400
        },
        22: {
            translateX: 200,
            translateY: -400
        },
        21: {
            translateX: 100,
            translateY: -400
        },
        20: {
            translateX: 0,
            translateY: -400
        },

    }

    onMount(() => {
        const interval = setInterval(() => {
            state = policy[state]
            translateX = translations[state].translateX
            translateY = translations[state].translateY
        }, 500);

        return () => clearInterval(interval);
    })

    let translateX = 0
    let translateY = 0
</script>


<svg width="500" height="500" version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
    <g id="grid" fill="none" stroke="var(--text-color)" stroke-width=".998">
     <path d="m0.499 0.499h499v499h-499z"/>
     <path d="m100.3 0.499v499"/>
     <path d="m200.1 0.499v499"/>
     <path d="m299.9 0.499v499"/>
     <path d="m399.7 0.499v499"/>
     <path d="m0.499 100.3h499"/>
     <path d="m0.499 200.1h499"/>
     <path d="m0.499 299.9h499"/>
     <path d="m0.499 399.7h499"/>
    </g>
    <g class="obstacle" fill="none" stroke="var(--text-color)">
     <rect x="10.479" y="210.08" width="79.84" height="79.84" stroke-width=".998"/>
     <g stroke-width=".998px">
      <path d="m10.479 230.04 19.96-19.96"/>
      <path d="m10.479 250 39.92-39.92"/>
      <path d="m10.479 269.96 59.88-59.88"/>
      <path d="m10.479 289.92 79.84-79.84"/>
      <path d="m30.439 289.92 59.88-59.88"/>
      <path d="m50.399 289.92 39.92-39.92"/>
      <path d="m70.359 289.92 19.96-19.96"/>
     </g>
     <rect x="110.28" y="210.08" width="79.84" height="79.84" stroke-width=".998"/>
     <g stroke-width=".998px">
      <path d="m110.28 230.04 19.96-19.96"/>
      <path d="m110.28 250 39.92-39.92"/>
      <path d="m110.28 269.96 59.88-59.88"/>
      <path d="m110.28 289.92 79.84-79.84"/>
      <path d="m130.24 289.92 59.88-59.88"/>
      <path d="m150.2 289.92 39.92-39.92"/>
      <path d="m170.16 289.92 19.96-19.96"/>
     </g>
     <rect x="210.08" y="210.08" width="79.84" height="79.84" stroke-width=".998"/>
     <g stroke-width=".998px">
      <path d="m210.08 230.04 19.96-19.96"/>
      <path d="m210.08 250 39.92-39.92"/>
      <path d="m210.08 269.96 59.88-59.88"/>
      <path d="m210.08 289.92 79.84-79.84"/>
      <path d="m230.04 289.92 59.88-59.88"/>
      <path d="m250 289.92 39.92-39.92"/>
      <path d="m269.96 289.92 19.96-19.96"/>
     </g>
     <rect x="309.88" y="210.08" width="79.84" height="79.84" stroke-width=".998"/>
     <g stroke-width=".998px">
      <path d="m309.88 230.04 19.96-19.96"/>
      <path d="m309.88 250 39.92-39.92"/>
      <path d="m309.88 269.96 59.88-59.88"/>
      <path d="m309.88 289.92 79.84-79.84"/>
      <path d="m329.84 289.92 59.88-59.88"/>
      <path d="m349.8 289.92 39.92-39.92"/>
      <path d="m369.76 289.92 19.96-19.96"/>
     </g>
    </g>
    <g id="player" stroke="var(--text-color)" transform="translate({translateX} {translateY})">
     <circle id="player-outer" cx="50.399" cy="449.6" r="29.94" fill="none" stroke-width="1.996"/>
     <g id="player-inner" fill="none">
      <g stroke-width=".998">
       <path d="m40.419 469.56 29.94-29.94"/>
       <path d="m30.439 449.6 19.96-19.96"/>
       <path d="m30.439 459.58 29.94-29.94"/>
       <path d="m30.439 469.56 39.92-39.92"/>
       <path d="m50.399 469.56 19.96-19.96"/>
       <path d="m60.379 469.56 9.98-9.9801"/>
      </g>
      <path d="m30.439 439.62 9.98-9.9801" stroke-width=".998px"/>
     </g>
    </g>
    <g id="goal" fill="var(--text-color)">
     <path d="m77.947 76.355-27.544-50.572-0.838 1.5424-26.717 49.029zm-3.352-2.0565h-48.395l24.203-44.402z"/>
     <g fill="none" stroke="var(--text-color)" stroke-width=".998px">
      <path d="m40.419 50.399h19.96"/>
      <path d="m40.419 60.379h19.96"/>
      <path d="m30.439 70.359h39.92"/>
      <path d="m80.339 10.479-19.96 19.96"/>
      <path d="m70.359 50.399h19.96"/>
      <path d="m70.359 80.339 9.98 9.98"/>
      <path d="m30.439 80.339-9.98 9.98"/>
      <path d="m30.439 50.399h-19.96"/>
      <path d="m40.419 30.439-19.96-19.96"/>
     </g>
    </g>
</svg>
