import { defineConfig, devices } from '@playwright/test'

const PORT = Number(process.env.PW_PORT ?? 8765)   // PW_PORT=8000 runs against a monitor and demo shop that are already running (the Demo test needs the shop to report to it)
const REPLAY = !!process.env.REPLAY_ONLY   // replay.spec.ts drives the server's single replay, so it runs alone with one worker
const DEMO = !!process.env.DEMO_ONLY       // demo.spec.ts starts the real demo shop and breaks it four ways (about 15 minutes), also alone with one worker
const SINGLE = REPLAY || DEMO
// The FastAPI app serves the built frontend (app/frontend/dist) and the API on one port, exactly like the Docker image.
export default defineConfig({
  testDir: './tests',
  testMatch: REPLAY ? /replay\.spec\.ts/ : DEMO ? /demo\.spec\.ts/ : /.*\.spec\.ts/,
  testIgnore: SINGLE ? [] : /(replay|demo)\.spec\.ts/,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: !SINGLE,
  workers: SINGLE ? 1 : 4,
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
