import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const WIDTHS = [375, 768, 1280, 1920]
const PAGES: { path: string; h1: string; ready: RegExp }[] = [
  { path: '/', h1: 'Is the system healthy right now?', ready: /Replay of recorded AnoMod data/ },
  { path: '/how-it-works', h1: 'How it works', ready: /replay/i },
  { path: '/proofs', h1: 'Proofs', ready: /Run the checks again/ },
  { path: '/results', h1: 'Results', ready: /Fewest false alarms/ },
  { path: '/testbed', h1: 'Testbed', ready: /Read this first/ },
  { path: '/?tab=live', h1: 'Is our demo shop healthy right now?', ready: /Live demo of our own test shop/ },
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
    await expect(page.getByRole('link', { name: 'Proofs' }).first()).toHaveAttribute('aria-current', 'page')
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

  test('dark colour scheme changes the palette', async ({ browser }) => {
    const light = await (await browser.newContext({ colorScheme: 'light' })).newPage()
    const dark = await (await browser.newContext({ colorScheme: 'dark' })).newPage()
    await light.goto('/'); await dark.goto('/')
    await expect(light.getByRole('navigation', { name: 'Main' })).toBeVisible()
    await expect(dark.getByRole('navigation', { name: 'Main' })).toBeVisible()
    const bg = (p: typeof light) => p.evaluate(() => getComputedStyle(document.body).backgroundColor)
    expect(await bg(light)).not.toBe(await bg(dark))
    const a = await new AxeBuilder({ page: dark }).withTags(['wcag2aa']).analyze()
    expect(a.violations.map((v) => v.id)).toEqual([])
  })
})
