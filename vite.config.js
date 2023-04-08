import { sveltekit } from "@sveltejs/kit/vite";
import { imagetools } from "vite-imagetools";

/** @type {import('vite').UserConfig} */
const config = {
  plugins: [
    sveltekit(),
    imagetools({
      defaultDirectives: (url) => {
        return new URLSearchParams({
          format: "webp",
        });
      },
    }),
  ],
  optimizeDeps: {
    include: ["highlight.js", "highlight.js/lib/core"],
  },
};

export default config;
