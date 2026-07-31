import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

// These checks share one backend with the other specs, and several of those
// approve plans — which unlocks the mission clock and changes what is on screen.
// Start each check from a fresh awaiting-approval run so the audit is stable.
test.beforeEach(async ({ request }) => {
  await request.post('http://127.0.0.1:8000/api/optimization/run', {
    data: { scenario_id: 'nepal-national-demo', requested_by: 'e2e-a11y-precondition' },
    timeout: 180_000,
  });
});

const WORKSPACES = ['operations', 'evidence', 'math', 'review'];

for (const workspace of WORKSPACES) {
  test(`${workspace} workspace has no serious accessibility violations`, async ({ page }) => {
    await page.goto(`/#${workspace}`);
    await expect(page.getByText('Event stream connected')).toHaveCount(1, { timeout: 30_000 });
    await expect(page.locator('.ops-app')).toHaveAttribute('aria-busy', 'false', { timeout: 45_000 });

    const result = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();
    const serious = result.violations.filter(({ impact }) => impact === 'serious' || impact === 'critical');

    expect(
      serious,
      serious.map(({ id, help, nodes }) => `${id}: ${help} (${nodes.length} nodes)`).join('\n'),
    ).toEqual([]);
  });
}
