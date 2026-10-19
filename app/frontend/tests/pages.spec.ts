import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const WIDTHS = [375, 768, 1280, 1920]
const PAGES: { path: string; h1: string; ready: RegExp }[] = [
  { path: '/', h1: 'Can a 30-second window of telemetry tell us an API is failing?', ready: /How the results were produced/ },
  { path: '/predict', h1: 'Predict', ready: /Real test window/ },
  { path: '/models', h1: 'Models', ready: /Overfitting: before and after correction/ },
  { path: '/explain', h1: 'Explain', ready: /Explain this window/ },
  { path: '/data', h1: 'Data & method', ready: /Leakage checks and what they found/ },
  { path: '/limits', h1: 'Limits', ready: /Other things to keep in mind/ },
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
    await page.setViewportSize({ width: 768, height: 800 })
    await page.goto('/models')
    await expect(page.locator('table.rt tbody tr').first()).toBeVisible()
    expect(await page.locator('table.rt tbody tr').first().evaluate((e) => getComputedStyle(e).display)).toBe('block')
    await page.setViewportSize({ width: 375, height: 800 })
    await page.goto('/models')
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
    // keyboard order of the interactive controls on a form page
    await page.goto('/predict')
    await expect(page.getByRole('button', { name: 'Score' })).toBeVisible()
    await page.getByRole('button', { name: 'Binary' }).focus()
    await page.keyboard.press('Tab')
    expect(await page.evaluate(() => document.activeElement?.tagName)).toMatch(/BUTTON|SELECT/)
  })

  test('landmarks, current page and form labels', async ({ page }) => {
    await page.goto('/predict')
    await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible()
    await expect(page.getByRole('main')).toBeVisible()
    await expect(page.getByRole('link', { name: 'Predict' }).first()).toHaveAttribute('aria-current', 'page')
    await expect(page.getByLabel('Model')).toBeVisible()
    await expect(page.getByLabel('Real test window')).toBeVisible()
  })

  test('respects prefers-reduced-motion', async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: 'reduce' })
    const page = await ctx.newPage()
    await page.route('**/api/**', async (r) => { await new Promise((res) => setTimeout(res, 2500)); await r.continue() })
    await page.goto('/')
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
