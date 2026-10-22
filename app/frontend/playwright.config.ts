import { defineConfig, devices } from '@playwright/test'

const PORT = Number(process.env.PW_PORT ?? 8765)   // PW_PORT=8000 runs against a monitor that is already running
const REPLAY = !!process.env.REPLAY_ONLY    // replay.spec.ts drives the server's single replay, so it runs alone with one worker
const MONITOR = !!process.env.MONITOR_ONLY  // monitor.spec.ts starts the real shop from the monitor page and breaks it four ways on the presenter page (about 30 minutes), alone with one worker
const SHOP = !!process.env.SHOP_ONLY        // storefront.spec.ts needs the real shop and its storefront: the launcher starts them (port 5000)
const SINGLE = REPLAY || MONITOR || SHOP
const ONLY = REPLAY ? /replay\.spec\.ts/ : MONITOR ? /monitor\.spec\.ts/ : SHOP ? /storefront\.spec\.ts/ : /.*\.spec\.ts/

// The FastAPI app serves the built frontend (app/frontend/dist) and the API on one port, exactly like the Docker image.
export default defineConfig({
  testDir: './tests',
  testMatch: ONLY,
  testIgnore: SINGLE ? [] : /(replay|monitor|storefront)\.spec\.ts/,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: !REPLAY && !MONITOR,
  workers: REPLAY || MONITOR ? 1 : 4,
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'tests/last-run.json' }]],
  use: { baseURL: SHOP ? 'http://127.0.0.1:5000' : `http://127.0.0.1:${PORT}`, trace: 'off', screenshot: 'only-on-failure' },
  webServer: SHOP
    ? { command: 'python -m testbed.launcher --no-browser', cwd: '../..', url: 'http://127.0.0.1:5000/healthz', reuseExistingServer: true, timeout: 300_000 }
    : { command: 'python -m app.backend.serve', env: { PORT: String(PORT) }, cwd: '../..', url: `http://127.0.0.1:${PORT}/api/health`, reuseExistingServer: true, timeout: 240_000 },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    // headless Firefox on Windows occasionally loses its software compositor under parallel load (RenderCompositorSWGL error); one retry, reported as flaky if used
    { name: 'firefox', retries: 1, use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
})
