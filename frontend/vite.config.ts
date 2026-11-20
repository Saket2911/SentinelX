import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The preview proxy host is supplied per-VM via env so a deployment-specific
// hostname never has to be committed.
const allowedHosts = process.env.__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS
  ? process.env.__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS.split(",")
  : [];

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    // PORT lets a hosting environment pin its own reserved port.
    port: Number(process.env.PORT ?? 5173),
    strictPort: true,
    allowedHosts,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
