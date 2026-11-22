/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

// In development the app calls /api/* and Vite forwards it to the API (no CORS needed).
// RIDESAFE_API_PROXY changes the target, e.g. RIDESAFE_API_PROXY=http://localhost:8765 npm run dev
const apiTarget = process.env.RIDESAFE_API_PROXY ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: {
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true, rewrite: (path) => path.replace(/^\/api/, '') },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => (id.includes('recharts') || id.includes('d3-') ? 'charts' : undefined),
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
