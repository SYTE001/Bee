/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        forge: {
          bg: "#0b0b12",
          panel: "#14141f",
          panel2: "#1b1b2b",
          border: "#2a2a40",
          accent: "#7c3aed",
          accent2: "#ec4899",
          text: "#e7e7f0",
          muted: "#9a9ab0",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
};
