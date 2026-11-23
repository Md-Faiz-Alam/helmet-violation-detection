/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv, type Plugin } from 'vite'

// In development the app calls /api/* and Vite forwards it to the API (no CORS needed).
// RIDESAFE_API_PROXY changes the target, e.g. RIDESAFE_API_PROXY=http://localhost:8765 npm run dev
const apiTarget = process.env.RIDESAFE_API_PROXY ?? 'http://localhost:8000'

/** Content-Security-Policy for the deployed site: scripts and styles only from the site itself (no inline
 * scripts), and network, image and video requests only to the site and the API. */
function contentSecurityPolicy(apiOrigin: string) {
  const api = apiOrigin ? ` ${apiOrigin}` : ''
  return [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    `connect-src 'self'${api}`,
    `img-src 'self' blob: data:${api}`,
    `media-src 'self' blob:${api}`,
    "font-src 'self' data:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join('; ')
}

function apiOriginOf(url: string | undefined) {
  return url && /^https?:\/\//.test(url) ? new URL(url).origin : ''
}

/** Writes the policy into the built Static Web Apps config, with the API origin of this build (VITE_API_URL). */
function cspHeader(apiOrigin: string): Plugin {
  let outDir = 'dist'
  return {
    name: 'ridesafe-csp',
    apply: 'build',
    configResolved(config) {
      outDir = config.build.outDir
    },
    closeBundle() {
      const file = `${outDir}/staticwebapp.config.json`
      const cfg = JSON.parse(readFileSync(file, 'utf8'))
      cfg.globalHeaders = { ...cfg.globalHeaders, 'Content-Security-Policy': contentSecurityPolicy(apiOrigin) }
      writeFileSync(file, `${JSON.stringify(cfg, null, 2)}\n`)
    },
  }
}

export default defineConfig(({ mode }) => {
  const apiOrigin = apiOriginOf(loadEnv(mode, process.cwd(), 'VITE_').VITE_API_URL)
  return {
    plugins: [react(), tailwindcss(), cspHeader(apiOrigin)],
    // `vite preview` sends the deployed policy too, so the end-to-end tests run under it (frontend/e2e)
    preview: { headers: { 'Content-Security-Policy': contentSecurityPolicy(apiOrigin) } },
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
      exclude: ['e2e/**', 'node_modules/**'],
    },
  }
})
