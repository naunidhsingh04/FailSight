// Runs Lighthouse (mobile and desktop) on every page of the built app, served by the FastAPI backend, using Playwright's Chromium.
// Usage (backend + built frontend on port 8765, e.g. `python -m uvicorn app.backend.main:app --port 8765` from the repo root):
//   node tests/lighthouse.mjs            -> writes tests/lighthouse-summary.json, exits 1 if any accessibility score is below 95
import { chromium } from '@playwright/test'
import { launch } from 'chrome-launcher'
import lighthouse from 'lighthouse'
import desktopConfig from 'lighthouse/core/config/desktop-config.js'
import { writeFileSync } from 'node:fs'

const BASE = process.env.BASE_URL ?? 'http://127.0.0.1:8765'
const PAGES = ['/', '/predict', '/models', '/explain', '/data', '/limits']
const chromePath = chromium.executablePath()
const out = []
let failed = false

for (const factor of ['mobile', 'desktop']) {
  const chrome = await launch({ chromePath, chromeFlags: ['--headless=new', '--no-sandbox', '--disable-gpu'] })
  for (const path of PAGES) {
    const flags = { port: chrome.port, output: 'json', logLevel: 'error', onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'] }
    const config = factor === 'desktop' ? desktopConfig : undefined
    const r = await lighthouse(BASE + path, flags, config)
    const c = r.lhr.categories
    const row = {
      page: path, formFactor: factor,
      performance: Math.round(c.performance.score * 100), accessibility: Math.round(c.accessibility.score * 100),
      bestPractices: Math.round(c['best-practices'].score * 100), seo: Math.round(c.seo.score * 100),
      failedAccessibilityAudits: Object.values(r.lhr.audits).filter((a) => c.accessibility.auditRefs.some((x) => x.id === a.id && x.weight > 0) && a.score === 0).map((a) => a.id),
    }
    if (row.accessibility < 95) failed = true
    out.push(row)
    console.log(`${factor.padEnd(7)} ${path.padEnd(9)} perf ${row.performance}  a11y ${row.accessibility}  best-practices ${row.bestPractices}  seo ${row.seo}${row.failedAccessibilityAudits.length ? '  FAILED: ' + row.failedAccessibilityAudits.join(',') : ''}`)
  }
  await chrome.kill()
}
writeFileSync(new URL('./lighthouse-summary.json', import.meta.url), JSON.stringify({ tool: 'lighthouse', base: BASE, results: out }, null, 1))
if (failed) { console.error('accessibility score below 95 on at least one page'); process.exit(1) }
