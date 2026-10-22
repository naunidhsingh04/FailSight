import { expect, test } from '@playwright/test'

const FOLD_NOTE = 'Original AnoMod models: 10-fold. Hybrids and LO2 work: 5-fold, chosen because of the time limit. All AnoMod models are scored on the same 647 test windows.'

test.describe('hybrid models page', () => {
  test('shows ten cards (five per data set), the fold note, the no-transfer note and range charts', async ({ page }) => {
    await page.goto('/examiners/hybrids')
    await expect(page.getByRole('heading', { level: 1, name: 'Hybrid models' })).toBeVisible()
    await expect(page.getByTestId('fold-note')).toHaveText(FOLD_NOTE)
    await expect(page.locator('[data-testid^="hybrid-anomod-"]')).toHaveCount(5)
    await expect(page.locator('[data-testid^="hybrid-lo2-"]')).toHaveCount(5)
    await expect(page.getByTestId('no-transfer')).toContainText(/transfer/i)
    await expect(page.getByTestId('verdict-anomod')).toContainText(/best/i)
    expect(await page.locator('svg.chart').count()).toBeGreaterThanOrEqual(2)
    await expect(page.getByText(/Unreliable: very small sample/).first()).toBeVisible()
    await expect(page.getByText(/95% range/).first()).toBeVisible()
  })
  test('is linked from the Proof page and from the examiners menu', async ({ page }) => {
    await page.goto('/proofs')
    await page.getByRole('link', { name: 'Hybrid models' }).first().click()
    await expect(page).toHaveURL(/examiners\/hybrids/)
    await page.goto('/examiners')
    await expect(page.getByRole('link', { name: /Hybrid models/ }).first()).toBeVisible()
  })
  test('the examiners page has the Colab box with the exact link', async ({ page }) => {
    await page.goto('/examiners')
    const box = page.getByTestId('colab-box')
    await expect(box).toContainText('Run it yourself in Colab')
    await expect(box.getByRole('link')).toHaveAttribute('href', 'https://colab.research.google.com/github/naunidhsingh04/FailSight/blob/main/notebooks/FailSight_Workflow.ipynb')
  })
})

test.describe('results page', () => {
  test.beforeEach(async ({ page }) => { await page.addInitScript(() => { try { localStorage.setItem('failsight.mode', 'detailed') } catch { /* ignore */ } }) })
  test('shows the top 9 models by default, a Show all button reveals every row, hybrids are tagged, and the LO2 tab works', async ({ page }) => {
    await page.goto('/examiners/results')
    await expect(page.getByTestId('top-note')).toContainText(/top 9/)
    await expect(page.getByTestId('top-note')).toContainText(/ranking uses the test set/)
    const rows = page.locator('table.rt').first().locator('tbody tr')
    await expect(rows).toHaveCount(9)
    await page.getByTestId('show-all').click()
    expect(await rows.count()).toBeGreaterThan(9)
    await expect(page.getByText('Hybrid neural network (two learning parts)').first()).toBeVisible()
    await page.getByRole('button', { name: /LO2 slice/ }).click()
    await expect(page.getByText(/Unreliable: very small sample/).first()).toBeVisible()
    await expect(page.getByText(/nothing about transfer|say nothing about transfer/i).first()).toBeVisible()
  })
  test('plain column headers with the technical name underneath', async ({ page }) => {
    await page.goto('/examiners/results')
    for (const h of ['Overall score', 'Healthy windows left alone', 'Fairness across both classes', 'Ranks risky above safe', 'Catches problems cleanly']) await expect(page.getByText(h).first()).toBeVisible()
    await expect(page.getByText('Decision-tree model').first()).toBeVisible()
  })
})

test('limits page states the hybrid and LO2 limits', async ({ page }) => {
  await page.goto('/examiners/limits')
  await expect(page.getByTestId('hybrid-limits')).toContainText(/5-fold/)
  await expect(page.getByTestId('hybrid-limits')).toContainText(/No forecasting/)
  await expect(page.getByTestId('hybrid-limits')).toContainText(/No transfer claim/)
})

