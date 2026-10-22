import { expect, test } from '@playwright/test'

for (const width of [375, 768, 1280, 1920]) {
  test.describe(`user flows at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } })

    test('Simple/Detailed toggle changes the page and is remembered', async ({ page }) => {
      await page.goto('/results')
      await expect(page.getByRole('button', { name: 'Simple' })).toHaveAttribute('aria-pressed', 'true')
      await expect(page.getByText('Fewest false alarms')).toBeVisible()
      await page.getByRole('button', { name: 'Detailed' }).click()
      await expect(page.getByRole('button', { name: 'Detailed' })).toHaveAttribute('aria-pressed', 'true')
      await expect(page.getByRole('heading', { name: 'Leaderboard' })).toBeVisible()
      await expect(page.getByText('Fewest false alarms')).toHaveCount(0)
      await page.reload()
      await expect(page.getByRole('heading', { name: 'Leaderboard' })).toBeVisible()
    })

    test('Results: three cards computed from test_metrics.csv; detailed leaderboard has intervals', async ({ page, request }) => {
      await page.goto('/results')
      for (const t of ['Best at catching failures', 'Fewest false alarms', 'Most balanced']) await expect(page.getByRole('heading', { name: t })).toBeVisible()
      const ms = (await (await request.get('/api/models')).json()).models.filter((m: any) => m.label_version === 'label_binary')
      const bestBA = Math.max(...ms.map((m: any) => m.metrics['official/ALL'].balanced_accuracy))
      await expect(page.getByTestId('card-bal')).toContainText(`${(bestBA * 100).toFixed(1)}%`)
      await page.getByRole('button', { name: 'Detailed' }).click()
      await expect(page.getByRole('table').first()).toBeVisible()
      await expect(page.getByText(/95% CI/).first()).toBeVisible()
    })

    test('Limits: states the weaknesses plainly in both modes', async ({ page }) => {
      await page.goto('/limits')
      await expect(page.getByText('55.6%').first()).toBeVisible()
      await expect(page.getByText(/Only 3 of the 124/)).toBeVisible()
      await expect(page.getByText(/The live feed is a replay/)).toBeVisible()
      await expect(page.getByText(/does not predict a future failure/)).toBeVisible()
      await expect(page.getByText(/relies on the system readings/)).toBeVisible()
      await page.getByRole('button', { name: 'Detailed' }).click()
      await expect(page.getByText('0.556').first()).toBeVisible()
      await expect(page.getByText('3 of 124').first()).toBeVisible()
      await expect(page.getByText(/not evidence of detection/)).toBeVisible()
    })

    const LIMITS = ['One app', 'One machine', 'Synthetic traffic and injected faults', 'Small held-out test', 'Per-type results are unreliable', 'Transfer to other apps has not been shown']

    test('Live monitor: the three plain limits say the monitored app is our own fictional test shop, and link to the full limits', async ({ page }) => {
      await page.goto('/?tab=live')
      await expect(page.getByRole('heading', { level: 1, name: 'Live monitor' })).toBeVisible()
      for (const t of [/our own fictional test shop/, /12 test runs with 15 problems/, /Other apps are unproven/]) await expect(page.getByText(t).first()).toBeVisible()
      await page.getByRole('link', { name: 'Limits', exact: true }).last().click()
      await expect(page.getByRole('heading', { level: 1, name: 'Limits' })).toBeVisible()
      await expect(page).toHaveURL(/\/examiners\/limits/)
    })

    test('For examiners lists every sub-page and each one opens', async ({ page }) => {
      await page.goto('/examiners')
      for (const n of ['Results', 'Connect', 'Testbed', 'Limits', 'Glossary', 'Recorded benchmark data']) await expect(page.getByRole('link', { name: new RegExp(`^${n}`) }).first()).toBeVisible()
      await page.getByRole('link', { name: /^Recorded benchmark data/ }).first().click()
      await expect(page.getByText(/cannot prove fault detection/).first()).toBeVisible()
      await expect(page.getByRole('navigation', { name: 'For examiners' })).toBeVisible()
      await expect(page.getByRole('link', { name: 'For examiners' }).first()).toHaveAttribute('aria-current', 'page')
    })

    test('Proof page: one sentence, the three numbers read from the result files, honest limits, full list collapsed', async ({ page, request }) => {
      await page.goto('/proofs')
      await expect(page.locator('.proof-sentence')).toContainText(/of the 15 problems we planted/)
      const table = page.getByRole('table', { name: 'The three numbers to know' })
      await expect(table).toContainText('Score on test runs it never saw')
      await expect(table).toContainText('Problems caught')
      await expect(table).toContainText('False alarms per hour')
      const r = await (await request.get('/api/testbed/results')).json()
      const o = r.overall.find((x: any) => x.method === `${r.frozen.model} (${r.frozen.feature_set})`)
      await expect(table).toContainText(o.balanced_accuracy.toFixed(2))
      await expect(page.locator('.honest')).toContainText(/one app.*test runs.*other apps are unproven/i)
      await expect(page.locator('details.all-checks')).not.toHaveAttribute('open', '')
      await page.getByText('Show all the checks behind these numbers').click()
      await expect(page.getByRole('heading', { name: 'Run the checks again' })).toBeVisible()
    })

    test('Connect page: validation scope, three snippets with copy buttons, connection state and webhook form', async ({ page }) => {
      await page.goto('/connect')
      for (const t of LIMITS) await expect(page.getByText(t).first()).toBeVisible()
      await expect(page.getByText(/validated only on our own test shop/)).toBeVisible()
      for (const t of ['Python (ASGI middleware)', 'Node (Express middleware)', 'Any server (tail an nginx or app access log)']) await expect(page.getByRole('heading', { name: t })).toBeVisible()
      await expect(page.getByRole('button', { name: /Copy the Python/ })).toBeVisible()
      await page.getByLabel(/App id/).fill('shop')
      await expect(page.getByRole('status').filter({ hasText: /Learning normal behaviour|Live|Not connected/ }).first()).toBeVisible()
      await page.getByRole('button', { name: 'Detailed' }).click()
      await expect(page.getByText(/API key required/)).toBeVisible()
      await expect(page.getByLabel('Webhook address')).toBeVisible()
    })

    test('Testbed page: limits first, the three live demos exactly as recorded, detailed tables on demand', async ({ page, request }) => {
      await page.goto('/testbed')
      for (const t of LIMITS) await expect(page.getByText(t).first()).toBeVisible()
      const live = (await (await request.get('/api/testbed/results')).json()).live
      expect(live.length).toBe(3)
      const tbl = page.getByRole('table', { name: /Live demo results/ })
      await expect(tbl).toBeVisible()
      for (const r of live) await expect(tbl).toContainText(String(r.seed))
      await expect(page.getByText(/too little data to quote a rate/)).toBeVisible()
      await page.getByRole('button', { name: 'Detailed' }).click()
      await expect(page.getByRole('table', { name: /Held-out test runs/ })).toBeVisible()
      await expect(page.getByText(/at least 5 fault instances/)).toBeVisible()
    })

    test('Limits page includes what the testbed shows and the six limits', async ({ page }) => {
      await page.goto('/limits')
      await expect(page.getByRole('heading', { name: 'What the testbed shows, and what it does not' })).toBeVisible()
      for (const t of LIMITS) await expect(page.getByText(t).first()).toBeVisible()
      await expect(page.getByText(/any app is not supported/)).toBeVisible()
      await expect(page.getByText(/An outside benchmark \(RCAEval, testing only\)/)).toBeVisible()
    })

    test('Proofs: checklist, audit table with C3 warning and the C7 baseline note, downloads', async ({ page }) => {
      await page.goto('/proofs')
      await page.getByText('Show all the checks behind these numbers').click()
      await expect(page.getByText('Passed', { exact: true })).toBeVisible()
      await expect(page.getByText('P7', { exact: true }).first()).toBeVisible()
      await expect(page.getByText('P13', { exact: true }).first()).toBeVisible()
      for (const id of ['P12', 'P16', 'P17']) await expect(page.getByText(id, { exact: true }).first()).toBeVisible()
      await expect(page.getByText(/C3\s+Time \/ run overlap/)).toBeVisible()
      await expect(page.getByText(/recorded its baseline/)).toBeVisible()
      await expect(page.getByRole('link', { name: /Download proof report \(JSON\)/ })).toHaveAttribute('href', '/api/proofs/download.json')
      await page.getByRole('tab', { name: /Terminal output/ }).click()
      await expect(page.getByLabel(/exactly as the terminal prints it/)).toContainText('P7')
    })

    test('unknown route shows a not-found page', async ({ page }) => {
      await page.goto('/nope')
      await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible()
    })
  })
}

test('API error state: backend down shows an error with retry', async ({ page }) => {
  await page.route('**/api/models', (r) => r.abort())
  await page.goto('/results')
  await expect(page.getByRole('alert').first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'Try again' }).first()).toBeVisible()
})
