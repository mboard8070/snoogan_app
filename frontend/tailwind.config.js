/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        snoogans: {
          dark: '#0f172a',
          card: '#1e293b',
          accent: '#3b82f6',
          green: '#22c55e',
          red: '#ef4444',
        },
      },
    },
  },
  plugins: [],
}
