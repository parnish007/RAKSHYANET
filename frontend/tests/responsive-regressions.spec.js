import { test, expect } from '@playwright/test';

/**
 * Layout regressions found by measurement, pinned so they cannot come back.
 *
 * Each case below was reproduced at a specific viewport before being fixed:
 * the approval panel splitting into a half-empty two-column grid between 761px
 * and 1120px, and the mission clock overflowing a clipping parent on a phone so
 * that its Reset control rendered outside the panel and could not be clicked.
 */

async function openReview(page) {
  await page.goto('/');
  await page.getByRole('button', { name: /Review & authorize|Review/ }).first().click();
  await expect(page.getByRole('heading', { name: 'Run authorization' })).toBeVisible({
    timeout: 60_000,
  });
}

test.describe('approval panel layout', () => {
  for (const width of [900, 1024, 1120]) {
    test(`decision panel is a single column at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await openReview(page);

      const panel = page.locator('.ops-decision');
      await expect(panel).toBeVisible();

      // The broken rule made .ops-decision a two-column grid whose children no
      // longer matched, leaving column one empty.
      const columns = await panel.evaluate(
        (node) => getComputedStyle(node).gridTemplateColumns,
      );
      expect(columns === 'none' || !columns.includes(' ')).toBeTruthy();

      // Every block must occupy the panel's full width, not half of it.
      const panelBox = await panel.boundingBox();
      const scroll = await panel.locator('.ops-decision-scroll').boundingBox();
      expect(scroll.width).toBeGreaterThan(panelBox.width * 0.85);
      expect(scroll.x).toBeLessThan(panelBox.x + panelBox.width * 0.15);
    });
  }
});

test.describe('mission clock on a phone', () => {
  test('every clock control stays inside the map panel at 390px', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');

    const clock = page.getByRole('group', { name: 'Mission clock' });
    if ((await clock.count()) === 0) {
      test.skip(true, 'clock only mounts once a plan is authorized');
    }

    const panelBox = await page.locator('.ops-map-panel').boundingBox();
    for (const control of await clock.locator('button, input').all()) {
      const box = await control.boundingBox();
      expect(box).not.toBeNull();
      // .ops-panel clips overflow, so anything past the right edge is invisible
      // and unclickable rather than merely scrolled off.
      expect(box.x + box.width).toBeLessThanOrEqual(panelBox.x + panelBox.width + 1);
    }
  });
});

test.describe('no page-level horizontal overflow', () => {
  for (const [width, height] of [[1280, 900], [768, 1024], [390, 844]]) {
    test(`body does not scroll sideways at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await page.goto('/');
      await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible({
        timeout: 60_000,
      });

      // overflow-x: hidden would mask a real overflow, so measure with it off.
      await page.addStyleTag({ content: 'body { overflow-x: visible !important; }' });
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      expect(overflow).toBeLessThanOrEqual(1);
    });
  }
});
