import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend runs on :8000 in dev; production build is served by the API container.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
