import { defineConfig } from "vite";
import { createReadStream, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react-swc";
import mkcert from "vite-plugin-mkcert";
import tailwindcss from "@tailwindcss/vite";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Serve ONNX Runtime WASM files from node_modules in dev
// (Vite's SPA fallback returns HTML for .wasm requests, breaking WebAssembly.instantiate)
function serveOrtWasm() {
  const distDir = join(__dirname, "node_modules/onnxruntime-web/dist");
  return {
    name: "serve-ort-wasm",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const filename = req.url?.split("/").pop();
        if (filename?.startsWith("ort-wasm") && filename.endsWith(".wasm")) {
          const filepath = join(distDir, filename);
          if (existsSync(filepath)) {
            res.setHeader("Content-Type", "application/wasm");
            createReadStream(filepath).pipe(res);
            return;
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  base: "/lfm-2-5-webgpu/",
  plugins: [serveOrtWasm(), mkcert(), react(), tailwindcss()],
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
