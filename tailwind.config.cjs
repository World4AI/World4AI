/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: {
				'w4-dark': 'hsl(260, 5%, 20%)',
				'w4-red': '#FF683C',
				'w4-yellow': '#FAE49E',
				'w4-blue': '#4EB6D7',
				'w4-light-blue': '#E7F1F2',
			}
		},
	},
	plugins: [],
}