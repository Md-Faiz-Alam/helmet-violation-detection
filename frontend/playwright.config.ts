import { defineConfig, devices } from '@playwright/test'

/**
 * End-to-end tests: the production build, served by `vite preview` with the deployed Content-Security-Policy,
 * against a real API (the Docker image in CI). The API must already be running:
 *
 *   docker run -p 8000:8000 ridesafe-api          (or: uvicorn ridesafe.api:app --port 8000, with RIDESAFE_SAMPLE_DIR)
 *   npx playwright install chromium               (once)
 *   npm run e2e                                   (E2E_API_URL=http://localhost:8765 for another port)
 */
const apiUrl = process.env.E2E_API_URL ?? 'http://localhost:8000'

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://localhost:4173',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npm run build && npx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173',
    env: { VITE_API_URL: apiUrl },
    timeout: 180_000,
    reuseExistingServer: !process.env.CI,
  },
})

export { apiUrl }
