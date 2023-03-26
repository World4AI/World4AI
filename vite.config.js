import { sveltekit } from "@sveltejs/kit/vite";
import { ViteImageOptimizer } from "vite-plugin-image-optimizer";

/** @type {import('vite').UserConfig} */
const config = {
  plugins: [
    sveltekit(),
    ViteImageOptimizer({
      /* pass your config */
    }),
  ],
  optimizeDeps: {
    include: ["highlight.js", "highlight.js/lib/core"],
  },
  assetsInclude: ["**/*.ipynb"],
};

export default config;
