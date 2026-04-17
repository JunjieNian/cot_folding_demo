import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5031,
    host: "0.0.0.0",
  },
  base: "./",
  build: {
    outDir: "dist",
    copyPublicDir: false, // data/ is too large; symlink dist/data -> public/data instead
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("plotly.js") || id.includes("react-plotly.js")) {
            return "plotly";
          }
          if (id.includes("node_modules/react") || id.includes("node_modules/react-dom")) {
            return "react-vendor";
          }
        },
      },
    },
  },
});
