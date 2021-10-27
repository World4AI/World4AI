<script>
    import { scaleLinear } from 'd3-scale';
    import { draw, fade } from 'svelte/transition';
    import { onMount } from 'svelte'

    let visible = true;

    onMount(() => {
        const interval = setInterval(() => {
            visible = !visible;
        }, 2500);

        return () => clearInterval(interval);
    });

    const xTicks = [1980, 1990, 2000, 2010];

    let portfolio = [
        { x: 1980, y: 5 },
        { x: 1981, y: 5.1 },
        { x: 1982, y: 5.2 },
        { x: 1983, y: 5.02},
        { x: 1984, y: 5.5 },
        { x: 1985, y: 5.6 },
        { x: 1986, y: 5.3 },
        { x: 1987, y: 5.7 },
        { x: 1988, y: 6.2 },
        { x: 1989, y: 6 },
        { x: 1990, y: 5.8 },
        { x: 1991, y: 6.3 },
        { x: 1992, y: 6.4 },
        { x: 1993, y: 6.32 },
        { x: 1994, y: 6.2 },
        { x: 1995, y: 6.12 },
        { x: 1996, y: 6 },
        { x: 1997, y: 6.55 },
        { x: 1998, y: 6.43 },
        { x: 1999, y: 6.22 },
        { x: 2000, y: 6.7 },
        { x: 2001, y: 7.1 },
        { x: 2002, y: 7 },
        { x: 2003, y: 6.8 },
        { x: 2004, y: 7.2 },
        { x: 2005, y: 7.4 },
        { x: 2006, y: 7 },
        { x: 2007, y: 7.5 },
        { x: 2008, y: 7.22 },
        { x: 2009, y: 7.55 },
        { x: 2010, y: 7.9 },
        { x: 2011, y: 8.1 },
        { x: 2012, y: 8.0 },
        { x: 2013, y: 8.43 },
        { x: 2014, y: 8.99 },
        { x: 2015, y: 9.2 },
        { x: 2016, y: 8.7 },
        { x: 2017, y: 8.65 },
        { x: 2018, y: 8.0 },
        { x: 2019, y: 8.5 },
        { x: 2020, y: 8.55 },
    ]

    let portfolio2 = [
        { x: 1980, y: 9 },
        { x: 1981, y: 8.1 },
        { x: 1982, y: 8.2 },
        { x: 1983, y: 8.02},
        { x: 1984, y: 8.5 },
        { x: 1985, y: 8.6 },
        { x: 1986, y: 7.3 },
        { x: 1987, y: 7.7 },
        { x: 1988, y: 7.2 },
        { x: 1989, y: 7 },
        { x: 1990, y: 7.8 },
        { x: 1991, y: 7.3 },
        { x: 1992, y: 7.4 },
        { x: 1993, y: 6.32 },
        { x: 1994, y: 6.2 },
        { x: 1995, y: 6.12 },
        { x: 1996, y: 5 },
        { x: 1997, y: 5.55 },
        { x: 1998, y: 5.43 },
        { x: 1999, y: 4.22 },
        { x: 2000, y: 5.7 },
        { x: 2001, y: 6.1 },
        { x: 2002, y: 6 },
        { x: 2003, y: 6.8 },
        { x: 2004, y: 7.2 },
        { x: 2005, y: 7.4 },
        { x: 2006, y: 8 },
        { x: 2007, y: 6.5 },
        { x: 2008, y: 6.22 },
        { x: 2009, y: 6.55 },
        { x: 2010, y: 7.9 },
        { x: 2011, y: 7.1 },
        { x: 2012, y: 7.0 },
        { x: 2013, y: 7.43 },
        { x: 2014, y: 7.99 },
        { x: 2015, y: 8.2 },
        { x: 2016, y: 7.7 },
        { x: 2017, y: 7.65 },
        { x: 2018, y: 6.0 },
        { x: 2019, y: 6.5 },
        { x: 2020, y: 6.55 },
    ]

    let portfolio3 = [
        { x: 1980, y: 3 },
        { x: 1981, y: 3.1 },
        { x: 1982, y: 3.2 },
        { x: 1983, y: 3.02},
        { x: 1984, y: 3.5 },
        { x: 1985, y: 3.6 },
        { x: 1986, y: 3.3 },
        { x: 1987, y: 3.7 },
        { x: 1988, y: 3.2 },
        { x: 1989, y: 3 },
        { x: 1990, y: 3.8 },
        { x: 1991, y: 3.3 },
        { x: 1992, y: 3.4 },
        { x: 1993, y: 3.32 },
        { x: 1994, y: 3.2 },
        { x: 1995, y: 3.12 },
        { x: 1996, y: 3 },
        { x: 1997, y: 3.55 },
        { x: 1998, y: 3.43 },
        { x: 1999, y: 3.22 },
        { x: 2000, y: 3.7 },
        { x: 2001, y: 3.1 },
        { x: 2002, y: 3 },
        { x: 2003, y: 3.8 },
        { x: 2004, y: 2.2 },
        { x: 2005, y: 2.4 },
        { x: 2006, y: 2 },
        { x: 2007, y: 2.5 },
        { x: 2008, y: 2.22 },
        { x: 2009, y: 2.55 },
        { x: 2010, y: 2.9 },
        { x: 2011, y: 2.1 },
        { x: 2012, y: 2.0 },
        { x: 2013, y: 2.43 },
        { x: 2014, y: 2.99 },
        { x: 2015, y: 2.2 },
        { x: 2016, y: 2.7 },
        { x: 2017, y: 2.65 },
        { x: 2018, y: 2.0 },
        { x: 2019, y: 2.5 },
        { x: 2020, y: 2.55 },
    ]


    export let width = 500;
    export let height = 200;

    $: minX = portfolio[0].x;
	$: maxX = portfolio[portfolio.length - 1].x;

    $: xScale = scaleLinear().domain([minX, maxX]).range([0, width])
    $: yScale = scaleLinear().domain([0, 10]).range([height, 0])

	$: path = `M${portfolio.map(p => `${xScale(p.x)},${yScale(p.y)}`).join('L')}`;
    $: path2 = `M${portfolio2.map(p => `${xScale(p.x)},${yScale(p.y)}`).join('L')}`;
    $: path3 = `M${portfolio3.map(p => `${xScale(p.x)},${yScale(p.y)}`).join('L')}`;

</script>

<svg {width} {height}>
{#each xTicks as tick}
    <g transform="translate({xScale(tick)},{height})">
        <text stroke="none" fill="var(--text-color)">{tick}</text>
    </g>
{/each}
{#if visible}
    <g fill="none" stroke="var(--text-color)">
        <path in:draw="{1000}" out:fade d={path}></path>
        <path in:draw="{1000}" out:fade d={path2}></path>
        <path in:draw="{1000}" out:fade d={path3}></path>
    </g>
{/if}
</svg>