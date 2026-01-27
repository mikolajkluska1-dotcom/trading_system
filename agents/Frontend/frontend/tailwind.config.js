/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Space Grotesk', 'monospace'],
      },
      colors: {
        bg: {
          main: "#050505", // Absolute Black
          card: "#0F0F11", // Surface
        },
        accent: {
          blue: "#3b82f6", // Revolut Blue
          white: "#FFFFFF",
        },
        // Old Void palette compatibility (optional, keeping for safety)
        void: "#050505",
        glass: "rgba(255, 255, 255, 0.05)",
        "glass-border": "rgba(255, 255, 255, 0.1)",
      },
      boxShadow: {
        'glow': '0 0 20px rgba(59, 130, 246, 0.15)',
        'card': '0 8px 32px 0 rgba(0, 0, 0, 0.36)',
      },
      animation: {
        marquee: 'marquee 25s linear infinite',
      },
      keyframes: {
        marquee: {
          '0%': { transform: 'translateX(0%)' },
          '100%': { transform: 'translateX(-100%)' },
        },
      },
    },
  },
  plugins: [],
}
