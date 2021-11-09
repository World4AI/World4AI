<script>
    import {draw} from 'svelte/transition';
</script>

<svg width="500" height="500" version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
 <g fill="none" stroke="var(--text-color)">
  <g fill="none" opacity=0.3 id="bodies" stroke-width="1px">
   <path in:draw="{{duration:2000}}" id="upper-body" d="m128.23 118.18v-70.07s8.1316-39.597 118.07-39.597c109.93 0 123.68 43.629 123.68 43.629l2.8788 144.66s-19.388 48.772-101.05 48.772-157.52-16.059-157.52 49.553v74.546h-48.596s-58.209-3.4588-58.209-120.03 63.615-114.33 63.615-114.33h184.95v-16.991z"/>
   <path in:draw="{{duration:2000}}" id="lower-body" d="m371.77 381.82v70.07s-8.1316 39.597-118.07 39.597c-109.93 0-123.68-43.629-123.68-43.629l-2.8788-144.66s19.388-48.772 101.05-48.772 157.52 16.059 157.52-49.553v-74.546h48.596s58.209 3.4588 58.209 120.03c0 116.57-63.615 114.33-63.615 114.33h-184.95v16.991z"/>
  </g>
  <g id="eyes" stroke="none" stroke-linejoin="round">
   <ellipse fill="var(--main-color-1)" id="upper-eye" cx="182.41" cy="65.64" rx="23.401" ry="23.401" stroke-width="1.0016"/>
   <ellipse fill="var(--main-color-2)" id="lower-eye" transform="scale(-1)" cx="-317.59" cy="-434.36" rx="23.401" ry="23.401"/>
  </g>
 </g>
</svg>
