// The guided demo end to end against the real demo shop. It starts the shop (about 1 minute), waits for the learning period (about 2.5 minutes) and breaks the shop four ways,
// so it runs alone with one worker: `npm run test:demo`.
import { expect, test } from '@playwright/test'
import { JARGON } from './jargon'

test.describe.configure({ mode: 'serial' })
test.use({ viewport: { width: 1280, height: 900 } })

const CHOICES: { button: string; plainWords: RegExp }[] = [
  { button: 'Make pages slow', plainWords: /slow|longer|response|time|wait|react/i },
  { button: 'Make the database slow', plainWords: /slow|database|took|longer|response|time/i },
  { button: 'Make a page crash', plainWords: /error|instead of a page|slow|longer/i },
  { button: 'Make the server crash', plainWords: /error|slow|server|shop|memory|visitors|requests|response|time|processor/i },
]

test('the shop starts from the page and FailSight finishes learning', async ({ page }) => {
  test.setTimeout(420_000)
  await page.goto('/')
  await expect(page.getByRole('heading', { level: 2, name: /Start the demo shop/ })).toBeVisible()
  const start = page.getByRole('button', { name: 'Start the demo shop' })
  if (await start.isVisible({ timeout: 8_000 }).catch(() => false)) await start.click()
  await expect(page.getByText(/The shop is running/)).toBeVisible({ timeout: 180_000 })
  await expect(page.getByText(/knows what normal looks like/)).toBeVisible({ timeout: 300_000 })
  for (const c of CHOICES) await expect(page.getByRole('button', { name: c.button })).toBeEnabled()
})

for (const c of CHOICES) {
  test(`"${c.button}" ends with a plain-language answer, a timeline and a reason`, async ({ page, request }) => {
    test.setTimeout(650_000)
    await request.post('/api/testbed/clear')                         // no fault left over from an earlier test
    await page.goto('/')
    await expect(page.getByText(/knows what normal looks like/)).toBeVisible({ timeout: 60_000 })
    await expect(page.getByText('Everything looks normal')).toBeVisible({ timeout: 360_000 })   // calm again (after a crash the shop needs a few minutes to warm up again) after the previous break
    await page.getByRole('button', { name: c.button }).click()
    await expect(page.getByText(/You broke it at \d\d:\d\d:\d\d/).first()).toBeVisible()
    const found = page.getByText(/FailSight noticed at \d\d:\d\d:\d\d \(\d+ seconds later\)/)
    const missed = page.getByText(/FailSight did not raise an alert this time/)
    await expect(found.or(missed)).toBeVisible({ timeout: 200_000 })
    await expect(found, 'the fault should be noticed').toBeVisible()
    // the big sentence is always one of the plain ones (it may already be back to normal, because a crash is short)
    await expect(page.locator('.answer')).toHaveText(/Everything looks normal|Something is wrong|Keeping an eye on it|the whole shop is down/)
    await expect(page.getByText(/^Why:/)).toBeVisible()
    await expect(page.locator('.reason')).toContainText(c.plainWords)
    expect(await page.locator('.reason').innerText(), 'no codes or jargon in the reason').not.toMatch(/\dxx|_|p9\d|rss|shap/i)
    await expect(page.getByRole('img', { name: /Response time of the last/ })).toContainText('You broke it')
    const text = await page.locator('main').innerText()
    expect(text.match(JARGON), 'jargon on the Demo page').toBeNull()
    const fix = page.getByRole('button', { name: 'Fix it now' })
    if (await fix.isEnabled()) await fix.click()                  // a crash is over by itself, so the button may already be off
  })
}

test('Detailed mode adds numbers; Simple stays free of jargon', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Detailed' }).click()
  await expect(page.getByRole('heading', { name: 'Numbers' })).toBeVisible()
  await page.getByRole('button', { name: 'Simple' }).click()
  await expect(page.getByRole('heading', { name: 'Numbers' })).toHaveCount(0)
})
