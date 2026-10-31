import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8080" },
  },
  // 构建产物直接由 FastAPI 挂载：app/static
  build: { outDir: "../app/static", emptyOutDir: true },
});
