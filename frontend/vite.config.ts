import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.SENTINELX_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: Number(process.env.PORT ?? 5173),
    strictPort: false,
    allowedHosts: (process.env.VITE_ALLOWED_HOSTS ?? "")
      .split(",")
      .map((host: string) => host.trim())
      .filter(Boolean),
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: Number(process.env.PORT ?? 4173),
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
