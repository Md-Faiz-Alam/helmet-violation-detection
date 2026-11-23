import { expect, test, type Page } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { apiUrl } from '../playwright.config.ts'

const photo = fileURLToPath(new URL('../../samples/image1.jpg', import.meta.url))

/** Fails the test on anything the Content-Security-Policy blocked and on uncaught page errors. */
function watchForProblems(page: Page) {
  const problems: string[] = []
  page.on('console', (m) => {
    if (/Content Security Policy|Refused to/i.test(m.text())) problems.push(m.text())
  })
  page.on('pageerror', (e) => problems.push(`page error: ${e.message}`))
  return problems
}

test('a photo is checked and its heads without a helmet are shown', async ({ page }) => {
  const problems = watchForProblems(page)
  await page.goto('/analyze')
  await page.locator('input[type=file]').setInputFiles(photo)

  await expect(page.getByText(/without a helmet at the deployed threshold/)).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('img', { name: 'Detections' })).toBeVisible()
  await expect(page.locator('li', { hasText: 'No helmet' }).first()).toBeVisible()
  expect(problems).toEqual([])
})

test('the sample video plays with its riders and their snapshots', async ({ page, request }) => {
  const problems = watchForProblems(page)
  const jobs = await (await request.get(`${apiUrl}/jobs`)).json()
  const sample = jobs.find((j: { readonly: boolean }) => j.readonly)
  expect(sample, 'the API has no sample job (RIDESAFE_SAMPLE_DIR)').toBeTruthy()

  // Overview: the sample loops in the hero, streamed from the API
  await page.goto('/')
  const hero = page.locator('video').first()
  await expect(hero).toHaveAttribute('src', new RegExp(`/jobs/${sample.id}/video$`))
  await expect.poll(() => hero.evaluate((v: HTMLVideoElement) => v.readyState), { timeout: 30_000 }).toBeGreaterThan(0)

  // Job page: every logged violation has a snapshot, served by the API through a signed link
  await page.goto(`/analyze/jobs/${sample.id}`)
  const snapshot = page.getByRole('img', { name: /^Rider \d+/ }).first()
  await expect(snapshot).toBeVisible({ timeout: 30_000 })
  await expect.poll(() => snapshot.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0)
  expect(problems).toEqual([])
})
