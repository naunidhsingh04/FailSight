import { expect, test } from '@playwright/test'
import { Buffer } from 'node:buffer'

for (const width of [375, 1280]) {
  test.describe(`user flows at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } })

    test('Predict: score a real sample window and see verdict, probabilities, SHAP and completeness', async ({ page }) => {
      await page.goto('/predict')
      await page.getByLabel('Real test window').selectOption('sample-07')
      await page.getByRole('button', { name: 'Score' }).click()
      await expect(page.getByRole('heading', { name: 'Result' })).toBeVisible()
      await expect(page.getByText(/Decision rule: call/)).toBeVisible()
      await expect(page.getByRole('group', { name: 'Predicted probability per class' })).toBeVisible()
      await expect(page.getByText('Input completeness')).toBeVisible()
      await expect(page.getByRole('list', { name: 'SHAP per feature' })).toBeVisible()
    })

    test('Predict: five-class model with tuned class weights', async ({ page }) => {
      await page.goto('/predict')
      await page.getByRole('button', { name: 'Five-class' }).click()
      await page.getByLabel('Model').selectOption('LGBM_tuned')
      await page.getByRole('button', { name: 'Score' }).click()
      await expect(page.getByText(/per-class weights/)).toBeVisible()
      await expect(page.getByRole('group', { name: 'Predicted probability per class' }).getByText('Performance')).toBeVisible()
    })

    test('Predict: uploading a CSV scores several windows', async ({ page, request }) => {
      const s = (await (await request.get('/api/samples')).json()).windows.slice(0, 4)
      const cols = Object.keys(s[0].features)
      const csv = ['system,window_id,' + cols.join(','), ...s.map((w: any) => [w.system, w.window_id, ...cols.map((c) => w.features[c] ?? '')].join(','))].join('\n')
      await page.goto('/predict')
      await page.getByRole('button', { name: 'Upload CSV' }).click()
      await page.getByLabel(/CSV of windows/).setInputFiles({ name: 'windows.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) })
      await expect(page.getByText('4 window(s) read')).toBeVisible()
      await page.getByRole('button', { name: 'Score' }).click()
      await expect(page.getByText('4 windows scored')).toBeVisible()
    })

    test('Predict: a CSV with almost no features is flagged unreliable', async ({ page }) => {
      await page.goto('/predict')
      await page.getByRole('button', { name: 'Upload CSV' }).click()
      await page.getByLabel(/CSV of windows/).setInputFiles({ name: 'thin.csv', mimeType: 'text/csv', buffer: Buffer.from('system,log_line_count,log_error_count\nTT,120,3\n') })
      await page.getByRole('button', { name: 'Score' }).click()
      await expect(page.getByText('unreliable').first()).toBeVisible()
      await expect(page.getByText(/Do not rely on this verdict/)).toBeVisible()
      await expect(page.getByText(/LO2 external test/).first()).toBeVisible()
    })

    test('Predict: bad inputs give clear errors', async ({ page }) => {
      await page.goto('/predict')
      await page.getByRole('button', { name: 'Upload CSV' }).click()
      await page.getByLabel(/CSV of windows/).setInputFiles({ name: 'nosystem.csv', mimeType: 'text/csv', buffer: Buffer.from('log_line_count\n10\n') })
      await page.getByRole('button', { name: 'Score' }).click()
      await expect(page.getByRole('alert').filter({ hasText: 'not accepted' })).toContainText('system')
      await page.getByLabel(/CSV of windows/).setInputFiles({ name: 'empty.csv', mimeType: 'text/csv', buffer: Buffer.from('a,b\n') })
      await expect(page.getByRole('alert').filter({ hasText: 'no data rows' })).toBeVisible()
      await page.getByLabel(/CSV of windows/).setInputFiles({ name: 'big.csv', mimeType: 'text/csv', buffer: Buffer.alloc(5 * 1024 * 1024 + 10, 'a') })
      await expect(page.getByRole('alert').filter({ hasText: 'limit is 5 MB' })).toBeVisible()
    })

    test('Explain: SHAP and LIME side by side with hover details', async ({ page }) => {
      await page.goto('/explain')
      await page.getByRole('button', { name: 'Explain this window' }).click()
      await expect(page.getByRole('heading', { name: 'SHAP' })).toBeVisible()
      await expect(page.getByRole('heading', { name: 'LIME' })).toBeVisible()
      await expect(page.getByText(/share .* of their top 5 features/)).toBeVisible()
      const first = page.getByRole('list', { name: 'SHAP per feature' }).getByRole('listitem').first()
      await first.focus()
      await expect(first.locator('.fbar-tip')).toBeVisible()
      await expect(first.locator('.fbar-tip')).toContainText('group')
    })

    test('Models: toggle label, sort, inspect a model', async ({ page }) => {
      await page.goto('/models')
      await expect(page.getByRole('table').first()).toBeVisible()
      await page.getByRole('button', { name: 'Five-class' }).click()
      await expect(page.getByText(/five-class labels/).first()).toBeVisible()
      if (width >= 640) {
        await page.getByRole('button', { name: 'Normal recall (95% CI)' }).click()
        await expect(page.locator('th[aria-sort]').first()).toHaveAttribute('aria-sort', /ascending|descending/)
      }
      await page.getByRole('button', { name: 'Fusion', exact: true }).first().click()
      await expect(page.getByRole('heading', { name: /Fusion: confusion matrix and curves/ })).toBeVisible()
      await expect(page.getByRole('img', { name: /Confusion matrix of Fusion/ })).toBeVisible()
      await expect(page.getByRole('img', { name: /ROC curve of Fusion/ })).toBeVisible()
    })

    test('Limits: states the weaknesses plainly', async ({ page }) => {
      await page.goto('/limits')
      await expect(page.getByText('0.556').first()).toBeVisible()
      await expect(page.getByText('3 of 124').first()).toBeVisible()
      await expect(page.getByText(/not evidence of detection/)).toBeVisible()
      await expect(page.getByText(/28/).first()).toBeVisible()
    })

    test('Data & method: leakage findings and class counts', async ({ page }) => {
      await page.goto('/data')
      await expect(page.getByText(/All 26 runs appear in both train and test/)).toBeVisible()
      await expect(page.getByRole('img', { name: /before and after SMOTE, binary/ })).toBeVisible()
    })

    test('unknown route shows a not-found page', async ({ page }) => {
      await page.goto('/nope')
      await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible()
    })
  })
}

test('API error state: backend down shows an error with retry', async ({ page }) => {
  await page.route('**/api/data', (r) => r.abort())
  await page.goto('/')
  await expect(page.getByRole('alert').first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'Try again' }).first()).toBeVisible()
})
