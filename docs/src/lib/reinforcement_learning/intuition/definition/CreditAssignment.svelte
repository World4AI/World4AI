<script>
    import { onMount } from 'svelte';
    let offset = 0;

    onMount(() => {
        let interval = setInterval(() => {
            offset += 20;
        }, 1000);

        return () => clearInterval(interval);
    })
</script>

<svg width="400" height="100" version="1.1" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
     <marker id="TriangleOutM" overflow="visible" orient="auto">
      <path transform="scale(.4)"  d="m5.77 0-8.65 5v-10z" fill="context-stroke" fill-rule="evenodd" stroke="var(--text-color)" stroke-width="1pt"/>
     </marker>
    </defs>
    <g stroke="#000">
     <g id="rewards" stroke-linejoin="round">
      <g fill="#ff683c">
       <rect x="2.6028" y="4.0861" width="34.795" height="34.795"/>
       <rect x="42.807" y="4.2908" width="34.795" height="34.795"/>
       <rect x="82.807" y="4.2908" width="34.795" height="34.795"/>
       <rect x="122.81" y="4.0861" width="34.795" height="34.795"/>
       <rect x="162.81" y="4.0861" width="34.795" height="34.795"/>
       <rect x="202.6" y="4.0861" width="34.795" height="34.795"/>
       <rect x="242.6" y="4.0861" width="34.795" height="34.795"/>
       <rect x="282.81" y="4.0861" width="34.795" height="34.795"/>
       <rect x="322.6" y="4.0861" width="34.795" height="34.795"/>
      </g>
      <rect x="362.6" y="4.0861" width="34.795" height="34.795" fill="#4eb6d7"/>
     </g>
     <path d="m45.603 83.301 24.999 0.27196" fill="none" marker-end="url(#TriangleOutM)" stroke-width="1px"/>
     <g id="actions" stroke="var(--text-color)" fill="none" stroke-width="1px">
      <path d="m17.602 68.573v25" marker-end="url(#TriangleOutM)"/>
      <path d="m45.603 83.301 24.999 0.27196" marker-end="url(#TriangleOutM)"/>
      <path d="m87.602 83.573 24.999 0.27196" marker-end="url(#TriangleOutM)"/>
      <path d="m122.6 83.573 24.999 0.27196" marker-end="url(#TriangleOutM)"/>
      <path d="m177.6 68.573v25" marker-end="url(#TriangleOutM)"/>
      <path d="m217.6 68.573v25" marker-end="url(#TriangleOutM)"/>
      <path d="m275.44 84.086-24.999-0.27196" marker-end="url(#TriangleOutM)"/>
      <path d="m312.6 84.086-24.999-0.27196" marker-end="url(#TriangleOutM)"/>
      <path d="m352.6 84.086-24.999-0.27196" marker-end="url(#TriangleOutM)"/>
      <path d="m382.6 68.573v25" marker-end="url(#TriangleOutM)"/>
     </g>
     <path d="m57.602 64.086" fill="none"/>
     <g id="credit-assignment" stroke-dashoffset={offset} fill="none" stroke="var(--text-color)" stroke-dasharray="8, 2, 1, 2">
      <path d="m17.602 44.086v20.027"/>
      <path d="m52.635 44.127-31.217 20.929"/>
      <path d="m57.602 44.086v35"/>
      <path d="m107.6 44.086v35"/>
      <path d="m102.6 44.086-35 35"/>
      <path d="m92.602 44.086-70 25"/>
     </g>
    </g>
   </svg>