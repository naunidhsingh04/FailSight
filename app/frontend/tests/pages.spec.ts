import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { JARGON } from './jargon'

const WIDTHS = [375, 768, 1280, 1920]
const PAGES: { path: string; h1: string; ready: RegExp }[] = [
  { path: '/', h1: 'See FailSight catch a broken website', ready: /What does FailSight say\?/ },
  { path: '/how-it-works', h1: 'How it works', ready: /It checks every 10 seconds/ },
  { path: '/proofs', h1: 'Proof', ready: /Show all the checks behind these numbers/ },
  { path: '/examiners', h1: 'For examiners', ready: /Recorded benchmark data/ },
  { path: '/examiners/replay', h1: 'Recorded benchmark data', ready: /cannot prove fault detection/ },
  { path: '/examiners/results', h1: 'Results', ready: /Fewest false alarms/ },
  { path: '/examiners/connect', h1: 'Connect your app', ready: /Connection status/ },
  { path: '/examiners/testbed', h1: 'Testbed', ready: /Read this first/ },
  { path: '/examiners/limits', h1: 'Limits', ready: /It did not work on a different system/ },
  { path: '/examiners/glossary', h1: 'Glossary', ready: /Data leakage/ },
  // the addresses used before the redesign must keep working
  { path: '/?tab=live', h1: 'See FailSight catch a broken website', ready: /What does FailSight say\?/ },
  { path: '/?tab=replay', h1: 'Recorded benchmark data', ready: /cannot prove fault detection/ },
  { path: '/console', h1: 'Recorded benchmark data', ready: /cannot prove fault detection/ },
  { path: '/results', h1: 'Results', ready: /Fewest false alarms/ },
  { path: '/connect', h1: 'Connect your app', ready: /Connection status/ },
  { path: '/testbed', h1: 'Testbed', ready: /Read this first/ },
  { path: '/limits', h1: 'Limits', ready: /It did not work on a different system/ },
  { path: '/glossary', h1: 'Glossary', ready: /Data leakage/ },
]

for (const width of WIDTHS) {
  test.describe(`viewport ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } })
    for (const pg of PAGES) {
      test(`${pg.path}: renders, no overflow, no errors, no axe violations`, async ({ page }) => {
        const errors: string[] = []
        const failed: string[] = []
        page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
        page.on('pageerror', (e) => errors.push(String(e)))
        page.on('response', (r) => { if (r.status() >= 400) failed.push(`${r.status()} ${r.url()}`) })
        await page.goto(pg.path)
        await expect(page.getByRole('heading', { level: 1, name: pg.h1 })).toBeVisible()
        await expect(page.getByText(pg.ready).first()).toBeVisible()
        await page.waitForLoadState('networkidle')
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
        expect(overflow, 'horizontal scroll').toBeLessThanOrEqual(0)
        expect(errors, 'console errors').toEqual([])
        expect(failed, 'failed requests').toEqual([])
        const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
        expect(axe.violations.map((v) => `${v.id}: ${v.nodes.slice(0, 2).map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([])
      })
    }
  })
}

test.describe('responsive tables', () => {
  test('table becomes stacked cards under 900px and a real table above', async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('failsight.mode', 'detailed'))
    await page.setViewportSize({ width: 768, height: 800 })
    await page.goto('/results')
    await expect(page.locator('table.rt tbody tr').first()).toBeVisible()
    expect(await page.locator('table.rt tbody tr').first().evaluate((e) => getComputedStyle(e).display)).toBe('block')
    await page.setViewportSize({ width: 375, height: 800 })
    await page.goto('/results')
    const row = page.locator('table.rt tbody tr').first()
    await expect(row).toBeVisible()
    expect(await row.evaluate((e) => getComputedStyle(e).display)).toBe('block')
    expect(await page.locator('table.rt thead').first().evaluate((e) => getComputedStyle(e).position)).toBe('absolute')
    await page.setViewportSize({ width: 1280, height: 800 })
    expect(await row.evaluate((e) => getComputedStyle(e).display)).toBe('table-row')
  })
})

test.describe('accessibility behaviour', () => {
  test('skip link is the first tab stop and focus is visible', async ({ page, browserName }) => {
    await page.goto('/')
    await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible()
    const skip = page.getByRole('link', { name: 'Skip to content' })
    if (browserName === 'webkit') {
      // WebKit/Safari do not Tab onto links by default (a browser setting); focus the link directly and verify what the user would see
      await skip.focus()
    } else {
      await page.keyboard.press('Tab')
    }
    await expect(skip).toBeFocused()
    await expect(skip).toBeVisible()
    const box = await skip.boundingBox()
    expect(box!.y).toBeGreaterThanOrEqual(0)                      // the link slides into view when focused
    const ol = await skip.evaluate((e) => { const s = getComputedStyle(e); return { style: s.outlineStyle, width: parseFloat(s.outlineWidth) } })
    expect(ol.style, `outline on ${browserName}`).not.toBe('none')
    expect(ol.width).toBeGreaterThanOrEqual(2)
    // keyboard order: the mode toggle is followed by another focusable control
    await page.getByRole('button', { name: 'Simple' }).focus()
    await page.keyboard.press('Tab')
    expect(await page.evaluate(() => document.activeElement?.tagName)).toMatch(/BUTTON|A|SELECT/)
  })

  test('landmarks, current page and form labels', async ({ page }) => {
    await page.goto('/proofs')
    await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible()
    await expect(page.getByRole('main')).toBeVisible()
    await expect(page.getByRole('link', { name: 'Proof', exact: true }).first()).toHaveAttribute('aria-current', 'page')
    await page.getByText('Show all the checks behind these numbers').click()
    await expect(page.getByLabel('How much to run')).toBeVisible()
  })

  test('respects prefers-reduced-motion', async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: 'reduce' })
    const page = await ctx.newPage()
    await page.route('**/api/**', async (r) => { await new Promise((res) => setTimeout(res, 2500)); await r.continue() })
    await page.goto('/results')
    const sk = page.locator('.skeleton').first()
    await sk.waitFor({ state: 'attached' })
    expect(await sk.evaluate((e) => getComputedStyle(e).animationName)).toBe('none')
    await ctx.close()
  })

  test('light is the default even when the computer prefers dark; the button switches to dark and back', async ({ browser }) => {
    const ctx = await browser.newContext({ colorScheme: 'dark' })
    const page = await ctx.newPage()
    await page.goto('/')
    await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible()
    const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor)
    const light = await bg()
    await page.getByRole('button', { name: 'Dark colours' }).click()
    expect(await bg()).not.toBe(light)
    const a = await new AxeBuilder({ page }).withTags(['wcag2aa']).analyze()
    expect(a.violations.map((v) => v.id)).toEqual([])
    await page.getByRole('button', { name: 'Light colours' }).click()
    expect(await bg()).toBe(light)
    await ctx.close()
  })
})

for (const width of WIDTHS) {
  test.describe(`home page at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } })

    test('opens on our own demo, shows three steps with one big button each, and no jargon', async ({ page }) => {
      await page.goto('/')
      await expect(page.getByRole('heading', { level: 2 }).filter({ hasText: /^1\s*Start the demo shop/ })).toBeVisible()
      await expect(page.getByRole('heading', { level: 2 }).filter({ hasText: /^2\s*Break something/ })).toBeVisible()
      await expect(page.getByRole('heading', { level: 2 }).filter({ hasText: /^3\s*What does FailSight say\?/ })).toBeVisible()
      for (const b of ['Make pages slow', 'Make the database slow', 'Make a page crash', 'Make the server crash', 'Surprise me']) await expect(page.getByRole('button', { name: b })).toBeVisible()
      await expect(page.getByRole('navigation', { name: 'Main' }).getByRole('link')).toHaveCount(4)   // Demo, How it works, Proof, For examiners
      await expect(page.getByText('Recorded benchmark data')).toHaveCount(0)               // the replay is never on the home page
      await page.waitForLoadState('networkidle')
      expect((await page.locator('main').innerText()).match(JARGON), 'jargon in Simple mode').toBeNull()
    })

    test('an empty state always says what to do, never just Waiting', async ({ page }) => {
      await page.route('**/api/testbed/faults', (r) => r.fulfill({ json: { active: [], reachable: false, error: 'the demo shop is not reachable' } }))
      await page.goto('/')
      await expect(page.getByText('Ready. Press Start.')).toBeVisible()
      await expect(page.getByRole('button', { name: 'Start the demo shop' })).toBeVisible()
      await expect(page.getByText(/Start the shop in step 1 first/)).toBeVisible()
      expect(await page.locator('main').innerText()).not.toMatch(/\bWaiting\b/)
    })
  })
}

test('the recorded benchmark page says "Ready. Press Start." when idle and never contradicts itself', async ({ page }) => {
  await page.goto('/examiners/replay')
  const status = page.getByRole('status').filter({ hasText: /Ready\. Press Start\./ })
  await expect(status).toBeVisible()
  expect(await status.innerText()).not.toMatch(/Not playing|Connecting/)
})

test('technical words explain themselves on tap', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 })
  await page.route('**/api/testbed/faults', (r) => r.fulfill({ json: { active: [], reachable: true } }))
  await page.route('**/api/testbed/state', (r) => r.fulfill({ json: { config: { frozen_at: 'x', model: 'RF', feature_set: 'both', n_features: 8, threshold: 0.3, learning_period_s: 150, alert_rule: '2 of last 3 windows' }, apps: { shop: { app: 'shop', anchor: 1, status: 'learning', idle: false, last_event_at: 1, learning: { done: false, progress: 0.4 }, last: null, counters: {}, history: [], alerts: [] } } } }))
  await page.route('**/api/testbed/windows**', (r) => r.fulfill({ json: [] }))
  await page.goto('/')
  await expect(page.getByText(/40% done/).first()).toBeVisible()
  const help = page.getByRole('button', { name: /What does .Learning what normal looks like. mean\?/ })
  await help.click()                // a click is what a tap becomes; the button toggles, so it also works without hover
  await expect(page.getByRole('tooltip')).toContainText(/two minutes/)
})
