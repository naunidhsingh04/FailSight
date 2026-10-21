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

    test('Live demo tab: states the six limits, offers real fault switches, and switches back to the replay', async ({ page }) => {
      await page.goto('/?tab=live')
      await expect(page.getByRole('heading', { level: 1, name: 'Is our demo shop healthy right now?' })).toBeVisible()
      for (const t of LIMITS) await expect(page.getByText(t).first()).toBeVisible()
      await expect(page.getByRole('button', { name: 'Slow database queries' })).toBeVisible()
      await expect(page.getByRole('group', { name: 'How bad' }).getByRole('button', { name: 'Mild' })).toBeVisible()
      await page.getByRole('tab', { name: 'Recorded data (replay)' }).click()
      await expect(page.getByRole('heading', { level: 1, name: 'Is the system healthy right now?' })).toBeVisible()
      await page.getByRole('tab', { name: 'Live demo app' }).click()
      await expect(page).toHaveURL(/tab=live/)
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
    })

    test('Proofs: checklist, audit table with C3 warning and the C7 baseline note, downloads', async ({ page }) => {
      await page.goto('/proofs')
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
