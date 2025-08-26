import type { Config } from "tailwindcss"

export default {
    content: [
        "./src/app/**/*.{js,ts,jsx,tsx}",
        "./src/components/**/*.{js,ts,jsx,tsx}"
    ],
    theme: {
        extend: {
            colors: {
                brand: {
                    blue: "#005496"
                }
            },
            boxShadow: {
                soft: "0 6px 24px rgba(0 , 0, 0, 0.08)"
            }
        }
    },
    plugins: [],
} satisfies Config