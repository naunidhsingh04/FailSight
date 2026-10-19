import { defineConfig, devices } from '@playwright/test'

const PORT = 8765
// The FastAPI app serves the built frontend (app/frontend/dist) and the API on one port, exactly like the Docker image.
export default defineConfig({
  testDir: './tests',
  testMatch: /.*\.spec\.ts/,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: true,
  workers: 4,
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'tests/last-run.json' }]],
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: 'off', screenshot: 'only-on-failure' },
  webServer: {
    command: 'python -m app.backend.serve',
    env: { PORT: String(PORT) },
    cwd: '../..',
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: true,
    timeout: 240_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    // headless Firefox on Windows occasionally loses its software compositor under parallel load (RenderCompositorSWGL error); one retry, reported as flaky if used
    { name: 'firefox', retries: 1, use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
})
