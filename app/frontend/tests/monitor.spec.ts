// The live monitor end to end with the real shop. The shop is NOT running when this starts: the first test presses "Start the shop" on the monitor page (which starts the shop,
// the storefront and the load generator), waits for the learning period (about 2.5 minutes), and the next tests break the shop four ways on the presenter page (port 5000) while the
// monitor page watches. The last test stops the shop again so that the next browser starts from a stopped shop too. Runs alone, one worker: `npm run test:monitor` (about 30 minutes).
import { expect, test } from '@playwright/test'
import type { APIRequestContext } from '@playwright/test'
import { JARGON } from './jargon'

test.describe.configure({ mode: 'serial' })
test.use({ viewport: { width: 1280, height: 900 } })

const SHOP = 'http://127.0.0.1:5000'
const CHOICES: { button: string; plainWords: RegExp }[] = [
  { button: 'Make pages slow', plainWords: /slow|longer|response|time|wait|react/i },
  { button: 'Make the database slow', plainWords: /slow|database|took|longer|response|time/i },
  { button: 'Make a page crash', plainWords: /error|instead of a page|slow|longer/i },
  { button: 'Make the server crash', plainWords: /error|slow|server|shop|memory|visitors|requests|response|time|processor/i },
]

async function shopIsUp(request: APIRequestContext) {
  try { return (await request.get('http://127.0.0.1:8802/health', { timeout: 3000 })).ok() } catch { return false }
}

/** wait until the monitor has been calm for 20 s in a row (after a crash the shop needs a few minutes to warm up again); waits on the monitor's own state, not on page text */
async function calm(request: APIRequestContext, timeoutMs = 480_000) {
  const t0 = Date.now()
  let good = 0
  while (Date.now() - t0 < timeoutMs) {
    try {
      const st = await (await request.get('/api/testbed/state')).json()
      const a = st.apps?.shop
      good = a && a.learning?.done && a.status === 'healthy' && !a.last?.alert ? good + 1 : 0
    } catch { good = 0 }
    if (good >= 4) return
    await new Promise((r) => setTimeout(r, 5000))
  }
  throw new Error('the monitor did not become calm within ' + timeoutMs / 1000 + ' s')
}

test('Start the shop from a stopped shop: the monitor page starts it, and FailSight finishes learning', async ({ page, request }) => {
  test.setTimeout(480_000)
  await request.post('/api/testbed/stop')                           // a shop left over from an earlier run, if this page started it
  await expect.poll(() => shopIsUp(request), { timeout: 30_000, message: 'a shop is still running on port 8802: stop it first' }).toBe(false)
  await page.goto('/')
  await expect(page.getByRole('heading', { level: 1, name: 'Live monitor' })).toBeVisible()
  await expect(page.getByText('The shop is not running.')).toBeVisible()
  await expect(page.getByText('Ready. Press Start.')).toBeVisible()
  await page.getByRole('button', { name: 'Start the shop' }).click()
  await expect(page.getByText(/The shop is running\./).first()).toBeVisible({ timeout: 180_000 })
  await expect(page.getByRole('link', { name: 'Open the shop' })).toHaveAttribute('href', `${SHOP}/`)
  // the storefront was started with it
  await expect.poll(async () => (await request.get(`${SHOP}/healthz`)).ok(), { timeout: 60_000 }).toBe(true)
  const shopTab = await page.context().newPage()
  await shopTab.goto(SHOP + '/')
  await expect(shopTab.getByRole('heading', { level: 1 })).toContainText('Coach journeys across Europe')
  await shopTab.close()
  await expect(page.getByText(/learning what normal looks like/i).first()).toBeVisible()
  await expect(page.getByText(/knows what normal looks like/)).toBeVisible({ timeout: 300_000 })
  await expect(page.getByRole('link', { name: 'Open the presenter controls' })).toHaveAttribute('href', `${SHOP}/ops`)
})

for (const c of CHOICES) {
  test(`"${c.button}" on the presenter page: the monitor says so in plain words, with a timeline and a reason`, async ({ page, request }) => {
    test.setTimeout(900_000)
    await request.post(`${SHOP}/ops/api/fix`)                         // nothing left over from an earlier test
    await calm(request)
    await page.goto('/')
    await expect(page.getByText('Everything looks normal')).toBeVisible()
    const ops = await page.context().newPage()
    await ops.goto(`${SHOP}/ops`)
    await ops.getByRole('button', { name: c.button }).click()
    await expect(ops.getByRole('status')).toContainText(/You broke it|not answering/, { timeout: 20_000 })
    await ops.close()
    await expect(page.getByText(/(A problem started|The shop stopped answering) at \d\d:\d\d:\d\d/).first()).toBeVisible({ timeout: 60_000 })
    const found = page.getByText(/FailSight noticed at \d\d:\d\d:\d\d \(\d+ seconds later\)/)
    const missed = page.getByText(/FailSight did not raise an alert this time/)
    await expect(found.or(missed)).toBeVisible({ timeout: 300_000 })
    await expect(found, 'the problem should be noticed').toBeVisible()
    await expect(page.locator('.answer')).toHaveText(/Everything looks normal|Something is wrong|Keeping an eye on it|the whole shop is down/)   // a crash is short, so it may already be back to normal
    await expect(page.getByText(/^Why:/)).toBeVisible()
    await expect(page.locator('.reason')).toContainText(c.plainWords)
    expect(await page.locator('.reason').innerText(), 'no codes or jargon in the reason').not.toMatch(/\dxx|_|p9\d|rss|shap/i)
    await expect(page.getByRole('img', { name: /Response time of the last/ })).toContainText('A problem started')
    const text = await page.locator('main').innerText()
    expect(text.match(JARGON), 'jargon on the monitor page').toBeNull()
    expect(text.match(/\bdemo\b/i), 'the word demo').toBeNull()
    expect(text.match(/\b(db_slow|exception_route|latency|cpu_burn)\b/), 'fault names belong on the presenter page only').toBeNull()
    await request.post(`${SHOP}/ops/api/fix`)
  })
}

test('Detailed mode adds numbers; Simple stays free of jargon', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Detailed' }).click()
  await expect(page.getByRole('heading', { name: 'Numbers' })).toBeVisible()
  await page.getByRole('button', { name: 'Simple' }).click()
  await expect(page.getByRole('heading', { name: 'Numbers' })).toHaveCount(0)
})

test('Stop the shop: the page says it is not running and offers Start again', async ({ page, request }) => {
  test.setTimeout(120_000)
  await request.post(`${SHOP}/ops/api/fix`)
  await page.goto('/')
  await expect(page.getByText(/knows what normal looks like/)).toBeVisible()
  await page.getByRole('button', { name: 'Stop the shop' }).click()
  await expect(page.getByText('The shop was stopped.')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText('The shop is not running.')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: 'Start the shop' })).toBeVisible()
  await expect.poll(() => shopIsUp(request), { timeout: 30_000 }).toBe(false)
})
