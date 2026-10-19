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

    test('Proofs: checklist, audit table with C3 warning and the C7 baseline note, downloads', async ({ page }) => {
      await page.goto('/proofs')
      await expect(page.getByText('Passed', { exact: true })).toBeVisible()
      await expect(page.getByText('P7', { exact: true }).first()).toBeVisible()
      await expect(page.getByText('P13', { exact: true }).first()).toBeVisible()
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
