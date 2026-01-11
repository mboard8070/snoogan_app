/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Quicksand', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      colors: {
        // Original Snoogans theme
        background: '#0a0a0a',
        surface: '#1a1a2e',
        text: '#e0e0e0',
        accent: '#00ff9d',
        // Button colors
        btn: {
          active: {
            bg: '#1e3a5f',      // lighter blue background
            text: '#f87171',    // reddish text (rose-400)
            border: '#f87171',  // matching border
          },
          inactive: {
            bg: '#1e3a5f',      // lighter blue background
            text: '#4ade80',    // greenish text (green-400)
            border: '#4ade80',  // matching border
          },
        },
        // Additional colors
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
