import { defineConfig } from 'vite';

/**
 * Cross-origin isolation headers are set in dev so ONNX Runtime can use
 * SharedArrayBuffer and run its wasm multi-threaded. Without them it falls back
 * to a single thread and still works, just slower -- see src/model/session.ts.
 * A static host needs the same two headers to get threading in production.
 */
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  build: {
    rollupOptions: { input: { main: 'index.html', parity: 'parity.html' } },
    target: 'es2022',
    // The model and the runtime wasm are the download; don't let anything else
    // sneak past unnoticed.
    chunkSizeWarningLimit: 700,
  },
  optimizeDeps: {
    // Pre-bundling rewrites the paths ORT uses to locate its own wasm.
    exclude: ['onnxruntime-web'],
  },
});
