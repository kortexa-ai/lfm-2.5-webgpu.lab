import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import mkcert from "vite-plugin-mkcert";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: "/lfm-2-5-webgpu/",
  plugins: [mkcert(), react(), tailwindcss()],
  build: {
    outDir: "./dist",
  },
  server: {
    host: "0.0.0.0",
    port: 8038,
    open: true,
  },
  preview: {
    host: "0.0.0.0",
    port: 8038,
  },
});
