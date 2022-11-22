import adapter from '@sveltejs/adapter-static';
import preprocess from "svelte-preprocess";


/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter({
			// default options are shown
			pages: 'build',
			assets: 'build',
			fallback: null,
			precompress: false
		}),
    	alias: {
    	  $notebooks: 'src/notebooks'
    	},
	},
  	preprocess: [
  	  preprocess({
  	    postcss: true,
  	  }),
  	],
};

export default config;

