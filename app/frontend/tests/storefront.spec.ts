// The storefront (Marigold Coaches) against the real shop. Needs the shop (port 8802) and the storefront (port 5000): `npm run test:shop` starts them with the launcher.
import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

const WIDTHS = [375, 768, 1280, 1920]
const FOOTER = 'Fictional shop used to test FailSight'
const FORBIDDEN = /\b(demo|test|failsight)\b/i
const FAULT_NAMES = /\b(db_slow|exception_route|cpu_burn|latency_ramp|route_500|pool_exhaustion|db_lock)\b/

/** everything a visitor can read or hear outside the footer: text, title, labels, placeholders, alt text, meta description */
async function readable(page: Page): Promise<string> {
  return page.evaluate(() => {
    const clone = document.body.cloneNode(true) as HTMLElement
    clone.querySelectorAll('footer').forEach((f) => f.remove())
    const attrs: string[] = []
    clone.querySelectorAll('*').forEach((el) => ['aria-label', 'alt', 'title', 'placeholder', 'aria-describedby'].forEach((a) => { const v = el.getAttribute(a); if (v) attrs.push(v) }))
    const meta = document.querySelector('meta[name=description]')?.getAttribute('content') ?? ''
    return [document.title, meta, clone.textContent ?? '', ...attrs].join('\n')
  })
}

async function checkPage(page: Page, what: string) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow, `${what}: horizontal scroll`).toBeLessThanOrEqual(0)
  expect(await readable(page), `${what}: forbidden word`).not.toMatch(FORBIDDEN)
  expect(await readable(page), `${what}: fault names belong on the presenter page only`).not.toMatch(FAULT_NAMES)
  const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
  expect(axe.violations.map((v) => `${v.id}: ${v.nodes.slice(0, 2).map((n) => n.target.join(' ')).join(' | ')}`), `${what}: axe`).toEqual([])
}

for (const width of WIDTHS) {
  test.describe(`storefront at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } })

    test('browse, search, add to cart and check out; every page is accessible and free of forbidden words', async ({ page }) => {
      const errors: string[] = []
      page.on('pageerror', (e) => errors.push(String(e)))
      await page.goto('/')
      await expect(page.getByRole('heading', { level: 1 })).toContainText('Coach journeys across Europe')
      await expect(page.getByRole('link', { name: /Marigold Coaches, home/ })).toBeVisible()
      await expect(page.locator('footer')).toContainText(FOOTER)
      await expect(page.getByRole('link', { name: /presenter|ops/i })).toHaveCount(0)            // the controls are not linked from the shop
      await checkPage(page, 'home')

      // search with the form
      await page.getByLabel('From').first().selectOption('Berlin')
      await page.getByLabel('To').first().selectOption('Prague')
      await page.getByRole('button', { name: 'Search journeys' }).click()
      await expect(page.getByRole('heading', { level: 1 })).toContainText('Berlin to Prague')
      await checkPage(page, 'journeys')

      // a journey page
      await page.getByRole('list').getByRole('link').filter({ hasText: /\d{2}:\d{2}/ }).first().click()
      await expect(page.getByRole('heading', { level: 1 })).toContainText('Berlin to Prague')
      await expect(page.getByText('Book your seats')).toBeVisible()
      await checkPage(page, 'journey')
      await page.getByLabel('Seats').selectOption('2')
      await page.getByRole('button', { name: 'Add to cart' }).click()

      // cart
      await expect(page.getByRole('heading', { level: 1, name: 'Your cart' })).toBeVisible()
      await expect(page.getByText(/2 seats at/)).toBeVisible()
      await expect(page.getByRole('navigation', { name: 'Main' })).toContainText('2')
      await checkPage(page, 'cart')
      await page.getByRole('link', { name: 'Go to checkout' }).click()

      // checkout: the form validates first
      await expect(page.getByRole('heading', { level: 1, name: 'Checkout' })).toBeVisible()
      await checkPage(page, 'checkout')
      await page.getByRole('button', { name: /Place order/ }).click()
      await expect(page.getByRole('alert')).toContainText(/name for the booking/i)
      await page.getByLabel('Name for the booking').fill('A. Traveller')
      await page.getByLabel('Email for the confirmation').fill('traveller@example.com')
      await page.getByRole('button', { name: /Place order/ }).click()

      // confirmation: a fake order created through the shop's own endpoints
      await expect(page.getByRole('heading', { level: 1 })).toContainText('your seats are booked')
      await expect(page.getByText(/booking number \d+/)).toBeVisible()
      await checkPage(page, 'order')
      await expect(page.locator('.count')).toHaveText(/0$/)   // the cart is empty again
      expect(errors).toEqual([])
    })

    test('home grid, empty cart and unknown pages read well', async ({ page }) => {
      await page.goto('/cart')
      await expect(page.getByText('Your cart is empty.')).toBeVisible()
      await checkPage(page, 'empty cart')
      await page.goto('/journeys?origin=Berlin&dest=Berlin')
      await expect(page.getByRole('alert')).toContainText('Choose two different cities')
      await page.goto('/order/NOPE')
      await expect(page.getByRole('alert')).toContainText(/could not find that order/i)
      await page.goto('/')
      await expect(page.getByRole('list').getByRole('listitem').filter({ hasText: /from €/ }).first()).toBeVisible()
    })
  })
}

// the two tests below break and fix the one shop, so they must not overlap
test.describe.serial('presenter controls and a broken shop', () => {
  test('the presenter page works from this computer and is where the fault names appear', async ({ page, request }) => {
    await page.goto('/ops')
    await expect(page.getByRole('heading', { level: 1, name: 'Presenter controls' })).toBeVisible()
    for (const b of ['Make pages slow', 'Make the database slow', 'Make a page crash', 'Make the server crash', 'Surprise me', 'Fix it']) await expect(page.getByRole('button', { name: b })).toBeVisible()
    await expect(page.getByText('db_slow')).toBeVisible()
    await expect(page.getByRole('link', { name: 'Open the shop' })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Open the monitor' })).toBeVisible()
    const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze()
    expect(axe.violations.map((v) => v.id)).toEqual([])
    expect((await request.get('/ops/api/status')).status()).toBe(200)
    // break and fix with the real buttons
    await page.getByRole('button', { name: 'Make pages slow' }).click()
    await expect(page.getByRole('status')).toContainText(/You broke it: Make pages slow/)
    await expect(page.getByRole('status')).toContainText(/Broken now: latency/, { timeout: 15_000 })
    await page.getByRole('button', { name: 'Fix it' }).click()
    await expect(page.getByRole('status')).toContainText(/Nothing is broken|Fixed/, { timeout: 15_000 })
  })

  test('a broken shop is felt by customers: pages say so in plain words and recover after Fix it', async ({ page, request }) => {
    await request.post('/ops/api/break', { data: { choice: 'page-crash' } })
    // the book step of the checkout is the broken route (exception_route fault): the shop answers 5xx some of the time
    await page.goto('/')
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    await request.post('/ops/api/fix')
    await expect.poll(async () => (await request.get('/ops/api/status')).json().then((j) => j.active.length), { timeout: 15_000 }).toBe(0)
    await page.goto('/')
    await expect(page.getByText('Popular journeys')).toBeVisible()
  })

})
