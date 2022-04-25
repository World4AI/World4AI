/** @type {import('@sveltejs/kit').Config} */

import adapter from '@sveltejs/adapter-static';
const config = {
	kit: {
        vite: {
            optimizeDeps: {
                include: ["highlight.js", "highlight.js/lib/core"],
            },
        },
        prerender: {
          default: true
        },
        paths: {
            base: '',
            assets: ''
        },      
        adapter: adapter({
            pages: 'build',  // path to public directory
            assets: 'build',  // path to public directory
            fallback: null
        })
    }
};

export default config;
