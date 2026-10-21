// These tests drive the one replay that the server owns, so they must not run in parallel: `npm run test:replay` (one worker).
import { expect, test } from '@playwright/test'

test.describe.configure({ mode: 'serial' })

for (const width of [375, 768, 1280, 1920]) {
  test.describe(`replay at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } })

    test('start, pause, an alert appears, its explanation opens, stop', async ({ page, request }) => {
      test.setTimeout(150_000)
      await request.post('/api/replay/control', { data: { stop: true } })
      await page.goto('/examiners/replay')
      await expect(page.getByText(/Recorded benchmark data \(cannot prove fault detection\)/).first()).toBeVisible()
      await page.getByRole('button', { name: 'Slow performance' }).click()
      await page.getByRole('button', { name: '60×' }).click()
      await page.getByRole('button', { name: 'Start' }).click()
      await expect(page.getByText(/Playing: Slow performance at 60×/)).toBeVisible()
      const first = page.getByRole('list', { name: /Alert feed/ }).getByRole('button').first()
      await expect(first).toBeVisible({ timeout: 90_000 })
      await page.getByRole('button', { name: 'Pause' }).click()
      await expect(page.getByText(/Paused: Slow performance/)).toBeVisible()
      await first.click()
      const dlg = page.getByRole('dialog', { name: 'Why this alert?' })
      await expect(dlg).toBeVisible()
      await expect(dlg).toContainText(/failure risk \d+%/)
      await expect(dlg).toContainText('What the recording says')
      await expect(dlg).toContainText('Data quality')
      await page.keyboard.press('Escape')
      await expect(dlg).toHaveCount(0)
      await page.getByRole('button', { name: 'Stop' }).click()
      await expect(page.getByText(/Ready\. Press Start\./)).toBeVisible()
    })

    if (width === 1280) {
      test('Proofs: Re-run the checks streams the terminal output', async ({ page }) => {
        test.setTimeout(300_000)
        await page.goto('/proofs')
        await page.getByText('Show all the checks behind these numbers').click()
        await page.getByRole('button', { name: 'Re-run the checks' }).click()
        const con = page.getByTestId('proof-console')
        await expect(con).toContainText(/P1/, { timeout: 120_000 })
        await expect(con).toContainText('[finished: exit code 0]', { timeout: 240_000 })
        await expect(page.getByRole('button', { name: 'Re-run the checks' })).toBeEnabled()
      })
    }
  })
}
