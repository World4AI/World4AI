/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js,svelte,ts}"],
  theme: {
    extend: {
      colors: {
        "w4ai-red": "#ff683c",
        "w4ai-yellow": "#fae49e",
        "w4ai-blue": "#4eb6d7",
        "w4ai-lightblue": "#e7f1f2",
      },
    },
  },
  plugins: [],
};
