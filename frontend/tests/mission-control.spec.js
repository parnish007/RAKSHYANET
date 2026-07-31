import { expect, test } from '@playwright/test';

function monitorRuntime(page) {
  const pageErrors = [];
  const consoleErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  page.on('console', (message) => {
    if (message.type() !== 'error') return;
    const source = message.location()?.url ?? '';
    const blockedExternalTile = (
      message.text().includes('ERR_NETWORK_ACCESS_DENIED')
      && !/127\.0\.0\.1|localhost/.test(source)
    );
    if (!blockedExternalTile) consoleErrors.push(message.text());
  });
  return { pageErrors, consoleErrors };
}

// Every test in this file drives the same backend process, and several of them
// mutate its run state — approving a plan, injecting a closure, submitting
// evidence. Without a reset the suite became order-dependent: whichever test ran
// after an approval saw an approved plan and failed assertions about the
// unapproved gate. Each test therefore starts from a fresh awaiting-approval run.
test.beforeEach(async ({ request }) => {
  await request.post('http://127.0.0.1:8000/api/optimization/run', {
    data: { scenario_id: 'nepal-national-demo', requested_by: 'e2e-precondition' },
    timeout: 180_000,
  });
});

async function waitForMission(page, hash = '#operations') {
  await page.goto(`/${hash}`);
  await expect(page.getByText('Event stream connected')).toHaveCount(1, { timeout: 30_000 });
  await expect(page.locator('.ops-app')).toHaveAttribute('aria-busy', 'false', { timeout: 45_000 });
}

test('task workspaces disclose one mission stage at a time', async ({ page }, testInfo) => {
  const runtime = monitorRuntime(page);
  await waitForMission(page);

  await expect(page.getByRole('heading', { name: 'See what happened, and where help is going' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Operations', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('heading', { name: 'Grounded report analysis' })).toHaveCount(0);
  await expect(page.getByRole('heading', { name: 'Plan mathematics and solver evidence' })).toHaveCount(0);

  await page.getByRole('button', { name: 'Gemma evidence' }).click();
  await expect(page).toHaveURL(/#evidence$/);
  await expect(page.getByRole('heading', { name: 'Grounded report analysis' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Gemma evidence' })).toHaveAttribute('aria-pressed', 'true');

  await page.getByRole('tab', { name: 'Values sent to math' }).click();
  await expect(page.getByText('Highest supported field')).toBeVisible();
  await expect(page.getByText('Bounded urgency delta')).toBeVisible();
  await expect(page.getByText(/Gemma affects ranking only through this bounded delta/)).toBeVisible();
  await expect(page.locator('.ops-handoff-formula').locator('b')).toHaveText([
    /^\d+\.\d{4}$/,
    /^\d+\.\d{4}$/,
    /^\+\d+\.\d{4}$/,
  ]);
  await expect(page.locator('.ops-handoff-grid article.selected em')).toContainText(/supported · selected maximum/i);

  await page.getByRole('tab', { name: /Evidence needs/ }).click();
  await expect(page.getByText('Questions Gemma will not guess')).toBeVisible();
  const addEvidenceFromQuestion = page.locator('.ops-follow-up-questions').getByRole('button', { name: /Answer with evidence:/ });
  if (await addEvidenceFromQuestion.count()) {
    await addEvidenceFromQuestion.first().click();
    await expect(page.getByRole('dialog', { name: 'Add evidence' })).toBeVisible();
    await expect(page.getByText('Question to answer')).toBeVisible();
    await expect(page.getByLabel('Evidence answer')).toBeVisible();
    await expect(page.getByRole('group', { name: /Supporting questions/ })).toBeVisible();
    await page.getByRole('dialog', { name: 'Add evidence' }).evaluate(async (element) => {
      await Promise.all(element.getAnimations().map((animation) => animation.finished));
    });
    await page.screenshot({ path: testInfo.outputPath('contextual-evidence-intake.png'), fullPage: true });
    await page.getByRole('button', { name: 'Close evidence intake' }).click();
    const discard = page.getByRole('button', { name: 'Discard draft' });
    if (await discard.count()) await discard.click();
  }

  const assign = page.getByRole('button', { name: 'Assign' }).first();
  if (await assign.count()) {
    await assign.click();
    const form = page.locator('.ops-disposition-form');
    const recordAssignment = form.getByRole('button', { name: 'Record assignment' });
    await expect(recordAssignment).toBeEnabled();
    await recordAssignment.click();
    await expect(form.getByRole('alert')).toContainText(/collection plan/i);
    await expect(form.getByLabel('Collection plan')).toBeFocused();
    await page.screenshot({ path: testInfo.outputPath('assignment-validation.png'), fullPage: true });
    await form.getByLabel('Collection plan').fill('District field desk will verify the evidence gap.');
    await recordAssignment.click();
    await expect(page.getByText(/ASSIGNED · Field coordination desk/).first()).toBeVisible();
    await page.getByRole('button', { name: 'Operations', exact: true }).click();
    await page.getByRole('button', { name: 'Gemma evidence', exact: true }).click();
    await page.getByRole('tab', { name: /Evidence needs/ }).click();
    await expect(page.getByText(/ASSIGNED · Field coordination desk/).first()).toBeVisible();
  }

  await page.getByRole('tab', { name: /^Sources \(\d+\)$/ }).click();
  await expect(page.getByText(/^\d+ records consulted$/)).toBeVisible();

  await page.getByRole('button', { name: 'Math lab' }).click();
  await expect(page).toHaveURL(/#math$/);
  await expect(page.getByRole('heading', { name: 'Plan mathematics and solver evidence' })).toBeVisible();
  await expect(page.getByText('Route-feasible VRP snapshot')).toBeVisible();
  await expect(page.getByText('Comparison candidate only.')).toHaveCount(0);

  await page.getByRole('tab', { name: 'Convergence' }).click();
  await expect(page.getByRole('img', { name: 'Allocation fixed-point convergence' })).toBeVisible();
  await expect(page.getByText(/Dimensionless maximum change normalized by village demand/)).toBeVisible();
  await expect(page.getByText(/residual .* tolerance/).first()).toBeVisible();
  await expect(page.getByText(/Exact zero is rendered at the 0.001 display floor/)).toBeVisible();
  await expect(page.getByText(/backend records \d+ SLSQP iterations/)).toBeVisible();
  await expect(page.getByRole('table').getByText('Within tolerance')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('math-convergence.png'), fullPage: true });

  await page.getByRole('tab', { name: 'Allocation comparison' }).click();
  await expect(page.getByText('Comparison candidate only.')).toBeVisible();
  await expect(page.getByText(/units are never summed/i)).toBeVisible();

  await page.getByRole('tab', { name: 'Validation & improvements' }).click();
  await expect(page.getByText('What needs improvement')).toBeVisible();

  await page.getByRole('button', { name: 'Review & authorize' }).click();
  await expect(page).toHaveURL(/#review$/);
  await expect(page.getByRole('heading', { name: 'Run authorization' })).toBeVisible();
  await expect(page.getByText(/Approval scope is national and explicit/)).toBeVisible();
  // With clearable warnings the primary action states the requirement instead of
  // routing to a diagnostics dialog, and the requirement panel spells out the two
  // steps. A hard block still routes to issue review.
  await expect(
    page.locator('.ops-decision-actions')
      .getByRole('button', { name: /Acknowledge below to authorize|Review \d+ issues?|Approve/ }),
  ).toBeVisible();
  await expect(page.locator('.ops-approval-requirement, .ops-override').first()).toBeVisible();

  expect(runtime.pageErrors).toEqual([]);
  expect(runtime.consoleErrors).toEqual([]);
});

test('map, camera, incident focus, and map-evidence workflow are complete', async ({ page }, testInfo) => {
  const runtime = monitorRuntime(page);
  await waitForMission(page);

  await expect(page.getByText(/3D terrain (active|fallback active)/)).toBeVisible({ timeout: 30_000 });
  const scene = page.locator('.terrain-scene');
  await expect(scene).toHaveAttribute('data-route-count', /\d+/);
  await expect(scene).toHaveAttribute('data-fleet-renderer', 'webp-symbol');

  const canvas = page.locator('.maplibregl-canvas');
  await expect(canvas).toBeVisible();
  await expect(scene).toHaveAttribute('data-vehicle-count', '9');
  await expect(scene).toHaveAttribute('data-air-vehicle-count', /\d+/);
  await expect(scene).toHaveAttribute('data-road-vehicle-count', /\d+/);
  await expect(scene).toHaveAttribute('data-infeasible-route-count', '0');
  await expect(scene).toHaveAttribute('data-motion-timeline', 'solver-stop-eta');
  await expect(scene).toHaveAttribute('data-timeline-checkpoint-count', /^[1-9]\d*$/);
  await expect(scene).toHaveAttribute('data-timeline-checkpoint-max-error', '0.000000');
  // Authorization gate: an unapproved plan must never animate. The fleet is
  // held at the depot and the mission clock stays pinned at zero.
  await expect(scene).toHaveAttribute('data-dispatch-active', 'false');
  await expect(scene).toHaveAttribute('data-mission-elapsed-minutes', '0.00');
  await expect(page.getByText('Fleet held at depot')).toBeVisible();
  await expect(page.getByRole('group', { name: 'Mission clock' })).toHaveCount(0);

  const heldPosition = await scene.getAttribute('data-lead-vehicle-position');
  await page.waitForTimeout(1_500);
  expect(await scene.getAttribute('data-lead-vehicle-position')).toBe(heldPosition);
  await expect(scene).toHaveAttribute('data-mission-elapsed-minutes', '0.00');

  const perspective = page.getByRole('button', { name: 'Show terrain perspective' });
  const fleet = page.getByRole('button', { name: 'Focus the moving fleet' });
  const topDown = page.getByRole('button', { name: 'Show top-down map' });
  await expect(scene).toHaveAttribute('data-camera-mode', 'incident');
  await expect(perspective).toHaveAttribute('aria-pressed', 'false');
  await fleet.click();
  await expect(fleet).toHaveAttribute('aria-pressed', 'true');
  await topDown.click();
  await expect(topDown).toHaveAttribute('aria-pressed', 'true');

  await page.getByRole('button', { name: /Jumla dirt road/i }).click();
  await expect(page.getByRole('heading', { name: 'Jumla' })).toBeVisible();
  await expect(scene).toHaveAttribute('data-focus-active', 'true');
  await expect(scene).toHaveAttribute('data-camera-mode', 'incident');
  await expect(topDown).toHaveAttribute('aria-pressed', 'false');
  // How many routes reach Jumla depends on the live Gemma analysis and which
  // village its evidence matched, so the exact count is not a UI invariant.
  // What must hold is that the summary renders against the real route count.
  await expect(
    page.getByText(/\d+ of \d+ feasible routes reach selection/),
  ).toBeVisible();
  if (await scene.getAttribute('data-terrain-status') === 'fallback') {
    await expect(scene).toHaveAttribute('data-camera-pitch', '0');
  }

  await page.getByRole('button', { name: 'Report map evidence' }).click();
  await expect(page.getByText('Choose the reported location on the map')).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByText('Choose the reported location on the map')).toHaveCount(0);

  await page.getByRole('button', { name: 'Report map evidence' }).click();
  await expect(scene).toHaveClass(/placing-incident/);
  const box = await canvas.boundingBox();
  await page.mouse.click(box.x + box.width * 0.4, box.y + box.height * 0.72);
  const intake = page.getByRole('dialog', { name: 'Add evidence' });
  await expect(intake).toBeVisible();
  await expect(intake.getByText('Closing: Map event report')).toBeVisible();
  await expect(intake.locator('.ops-gap-target small')).toContainText(/marker is evidence context, not a routable incident/i);
  const sourceName = `Map verification desk ${Date.now()}`;
  await intake.getByLabel('Source name', { exact: true }).fill(sourceName);
  await intake.getByRole('textbox', { name: 'Evidence answer' }).fill('Field desk reports a blocked mountain road near the selected map location.');
  await intake.getByRole('textbox', { name: /What exact village, ward, landmark, or coordinates/ }).fill('Sindhupalchok ward 6 near the selected map point.');
  await intake.getByText('Add optional facts & provenance').click();
  await intake.getByRole('textbox', { name: /^Reported place/ }).fill('Sindhupalchok ward 6');
  await intake.getByRole('textbox', { name: /^Casualties or injuries/ }).fill('3 injured, fatalities unverified');
  await intake.getByRole('combobox', { name: /^Damage level/ }).selectOption('major');
  await intake.getByRole('combobox', { name: /^Road access/ }).selectOption('blocked');
  await intake.getByRole('textbox', { name: /^Requested resources/ }).fill('medical kits and bridge assessment team');
  await intake.getByRole('button', { name: 'Queue source' }).click();
  await expect(intake.getByText(/map -?\d+\.\d{4}, -?\d+\.\d{4}/)).toBeVisible();

  const evidenceRequest = page.waitForRequest((request) =>
    request.url().endsWith('/api/gemma/analyze-submitted')
    && request.method() === 'POST');
  await intake.getByRole('button', { name: 'Analyze and recalculate plan' }).click();
  const submitted = (await evidenceRequest).postDataJSON();
  const mapEvidence = submitted.evidence.find((item) => item.source_name === sourceName);
  expect(mapEvidence.reported_latitude).toEqual(expect.any(Number));
  expect(mapEvidence.reported_longitude).toEqual(expect.any(Number));
  expect(mapEvidence.source_identifier).toMatch(/^map:\/\//);
  expect(mapEvidence.operator_context).toContain('Map event report');
  expect(mapEvidence.gap_target).toBe('Map event report');
  expect(mapEvidence.text).toContain('Operator-supplied structured facts:');
  expect(mapEvidence.text).toContain('What exact village, ward, landmark, or coordinates did the source verify? Answer: Sindhupalchok ward 6 near the selected map point.');
  expect(mapEvidence.text).toContain('Reported place: Sindhupalchok ward 6.');
  expect(mapEvidence.text).toContain('Casualty or injury report: 3 injured, fatalities unverified.');
  expect(mapEvidence.text).toContain('Reported damage level: major.');
  expect(mapEvidence.text).toContain('Reported road access: blocked.');
  expect(mapEvidence.text).toContain('Reported resource needs: medical kits and bridge assessment team.');
  await expect(intake).toHaveCount(0, { timeout: 45_000 });

  await page.screenshot({ path: testInfo.outputPath('operations-map.png'), fullPage: true });
  expect(runtime.pageErrors).toEqual([]);
  expect(runtime.consoleErrors).toEqual([]);
});

test('evidence can be queued and removed without losing mission state', async ({ page }) => {
  const runtime = monitorRuntime(page);
  await waitForMission(page, '#evidence');

  await page.getByRole('button', { name: 'Add evidence' }).first().click();
  const intake = page.getByRole('dialog', { name: 'Add evidence' });
  await expect(intake).toBeVisible();
  await intake.getByLabel('Source name', { exact: true }).fill('Sindhupalchok field desk');
  await intake.getByRole('textbox', { name: /^Report text/ }).fill('Field team reports one bridge blocked and verified medical access delays.');
  await intake.getByText('Add optional facts & provenance').click();
  await intake.getByRole('textbox', { name: /^Operator context/ }).fill('Confirm the blocked bridge before rerouting ground vehicles.');
  await intake.getByRole('combobox', { name: /^Source reliability/ }).selectOption('0.85');
  await intake.getByRole('button', { name: 'Queue source' }).click();
  await expect(intake.getByText('Sindhupalchok field desk')).toBeVisible();
  await expect(intake.getByText(/reliability 0.85/)).toBeVisible();
  await intake.getByRole('button', { name: 'Remove Sindhupalchok field desk' }).click();
  await expect(intake.getByText('Sindhupalchok field desk')).toHaveCount(0);
  await intake.getByLabel('Source name', { exact: true }).fill('Unqueued draft');
  await intake.getByRole('button', { name: 'Close evidence intake' }).click();
  await expect(intake.getByText('Discard the unqueued draft?')).toBeVisible();
  await intake.getByRole('button', { name: 'Keep editing' }).click();
  await intake.getByRole('button', { name: 'Close evidence intake' }).click();
  await intake.getByRole('button', { name: 'Discard draft' }).click();

  expect(runtime.pageErrors).toEqual([]);
  expect(runtime.consoleErrors).toEqual([]);
});

test('mock scenario deck switches baseline and road-block timelines', async ({ page }) => {
  const runtime = monitorRuntime(page);
  await waitForMission(page);

  const selector = page.getByRole('combobox', { name: 'Mock scenario' });
  await expect(selector.locator('option')).toHaveCount(5);
  await selector.selectOption('jumla-bridge-karnali-closure');
  await page.getByRole('button', { name: 'After road block' }).click();

  const activationResponse = page.waitForResponse((response) => (
    response.url().includes('/api/demo/scenarios/jumla-bridge-karnali-closure/activate')
    && response.request().method() === 'POST'
  ));
  await page.getByRole('button', { name: 'Load this scenario' }).click();
  const activated = await activationResponse;
  expect(activated.ok()).toBeTruthy();
  const payload = await activated.json();

  await expect(page.locator('.ops-app')).toHaveAttribute('aria-busy', 'false', { timeout: 90_000 });
  await expect(page.getByText('Active runtime')).toBeVisible();
  await expect(page.getByText(/karnali_pokhara_jumla closed/)).toBeVisible();
  expect(payload.run.parent_run_id).toBe(payload.baseline_run_id);
  expect(payload.run.blocked_edge_ids).toEqual(['karnali_pokhara_jumla']);
  expect(payload.run.route_feasible).toBe(true);

  expect(runtime.pageErrors).toEqual([]);
  expect(runtime.consoleErrors).toEqual([]);
});

test('a failed map-resource request exposes a scoped retry', async ({ page }) => {
  let villageRequests = 0;
  await page.route('**/api/villages', async (route) => {
    villageRequests += 1;
    if (villageRequests === 1) {
      await route.fulfill({ status: 503, body: 'temporary fixture failure' });
      return;
    }
    await route.continue();
  });

  await page.goto('/#operations');
  const retry = page.getByRole('button', { name: 'Retry map resources' });
  await expect(retry).toBeVisible({ timeout: 30_000 });
  await retry.click();
  await expect(page.getByRole('heading', { name: 'Active incidents' })).toBeVisible();
  await expect(page.locator('.ops-incident-row')).toHaveCount(8);
  await expect(retry).toHaveCount(0);
});

test('review actions, diagnostics, and trace remain human-controlled', async ({ page }) => {
  const runtime = monitorRuntime(page);
  await waitForMission(page, '#review');

  await page.getByRole('button', { name: /Review \d+ issues?/ }).click();
  const diagnostics = page.getByRole('dialog', { name: 'Math engine full diagnostics' });
  await expect(diagnostics).toBeVisible();
  await expect(diagnostics.getByRole('button', { name: 'Close full diagnostics' })).toBeFocused();
  await diagnostics.getByRole('button', { name: 'Overview' }).click();
  await expect(diagnostics.getByText('What was allocated')).toBeVisible();
  await expect(diagnostics.getByText(/declared units proposed/)).toHaveCount(0);
  await diagnostics.getByRole('button', { name: 'Resources' }).click();
  await expect(diagnostics.getByText('Available and assigned assets')).toBeVisible();
  await expect(diagnostics.getByText(/direct ETA .* projected tour/i).first()).toBeVisible();
  await diagnostics.getByRole('button', { name: 'Urgency' }).click();
  await expect(diagnostics.getByText('Urgency derivation')).toBeVisible();
  await diagnostics.getByRole('button', { name: 'Routes & closures' }).click();
  await expect(diagnostics.getByText('Close a corridor and force a new plan')).toBeVisible();
  await diagnostics.getByRole('button', { name: 'Close full diagnostics' }).click();

  await page.getByRole('button', { name: 'Request changes' }).click();
  const reviewDialog = page.getByRole('dialog', { name: 'Request changes to this plan' });
  await expect(reviewDialog).toBeVisible();
  await reviewDialog.getByLabel('Reason for requesting changes').fill('Verify the road closure and attach a field-source update.');
  await expect(reviewDialog.getByRole('button', { name: 'Submit request for changes' })).toBeEnabled();
  await expect(reviewDialog.getByText('Pinned run')).toBeVisible();

  const historyResponse = await page.request.get('http://127.0.0.1:8000/api/optimize/history');
  const [current] = await historyResponse.json();
  const newerResponse = await page.request.post('http://127.0.0.1:8000/api/optimization/run', {
    data: {
      scenario_id: current.scenario_id,
      analysis_id: current.analysis_id,
      requested_by: 'stale-review-test',
      parent_run_id: current.run_id,
      trigger: 'concurrency_test',
    },
  });
  expect(newerResponse.ok()).toBe(true);
  await expect(reviewDialog).toHaveCount(0, { timeout: 15_000 });
  await expect(page.getByRole('alert')).toContainText(/Authorization closed because a newer versioned run replaced/);

  await page.getByRole('button', { name: 'Request changes' }).click();
  const currentDialog = page.getByRole('dialog', { name: 'Request changes to this plan' });
  await currentDialog.getByLabel('Reason for requesting changes').fill('Rejecting the current snapshot until field evidence is verified.');
  await currentDialog.getByRole('button', { name: 'Submit request for changes' }).click();
  await expect(currentDialog).toHaveCount(0, { timeout: 15_000 });
  await expect(page.getByText('rejected', { exact: true }).first()).toBeVisible();

  await page.getByRole('button', { name: 'Inspect complete decision trace' }).click();
  const trace = page.getByRole('dialog', { name: 'Complete decision trace' });
  await expect(trace).toBeVisible();
  await expect(trace.getByText('Evidence consulted')).toBeVisible();
  await expect(trace.getByRole('button', { name: 'Close decision trace' })).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(trace).toHaveCount(0);

  expect(runtime.pageErrors).toEqual([]);
  expect(runtime.consoleErrors).toEqual([]);
});

test('an infeasible route snapshot cannot expose an approval action', async ({ page }) => {
  let expectedFeasibleCount = 0;
  const makeInfeasible = (run) => {
    expectedFeasibleCount = run.result.vrp_solution.routes.length - 1;
    run.route_feasible = false;
    run.approval_blockers = ['Injected infeasible route for authorization regression'];
    run.result.vrp_solution.routes[0].feasible = false;
    run.result.vrp_solution.routes[0].infeasibility_reason =
      'Injected infeasible route for authorization regression';
    return run;
  };
  await page.route('**/api/optimize/history', async (route) => {
    const response = await route.fetch();
    const history = await response.json();
    makeInfeasible(history[0]);
    await route.fulfill({ response, json: history });
  });
  await page.route('**/api/optimization/runs/*', async (route) => {
    if (route.request().method() !== 'GET') {
      await route.continue();
      return;
    }
    const response = await route.fetch();
    const run = makeInfeasible(await response.json());
    await route.fulfill({ response, json: run });
  });

  await waitForMission(page, '#review');
  await expect(page.getByText('Approval blocked')).toBeVisible();
  await expect(page.getByText(/Route plan is infeasible or empty/)).toBeVisible();
  await expect(page.getByRole('button', { name: /Review \d+ issues?/ })).toBeVisible();
  await expect(
    page.getByRole('button', { name: /Approve (demo plan|for coordination)/ }),
  ).toHaveCount(0);

  await page.getByRole('button', { name: 'Operations', exact: true }).click();
  await expect(page.getByText(/3D terrain (active|fallback active)/)).toBeVisible();
  const scene = page.locator('.terrain-scene');
  await expect(scene).toHaveAttribute('data-route-count', String(expectedFeasibleCount));
  await expect(scene).toHaveAttribute('data-infeasible-route-count', '1');
  await expect(scene).toHaveAttribute('data-vehicle-count', String(expectedFeasibleCount));
  await expect(page.getByText('1 route exception excluded')).toBeVisible();
  await expect(page.getByText('Route exception · not dispatched')).toBeVisible();
});

test('1366x768 laptop view has no page-level horizontal overflow', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  await waitForMission(page, '#operations');
  for (const workspace of ['Operations', 'Gemma evidence', 'Math lab', 'Review & authorize']) {
    await page.getByRole('button', { name: workspace, exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth === window.innerWidth)).toBe(true);
  }
  await page.getByRole('button', { name: 'Math lab', exact: true }).click();
  await page.getByRole('tab', { name: 'Convergence' }).click();
  await expect(page.getByRole('img', { name: 'Allocation fixed-point convergence' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('math-1366.png'), fullPage: true });
});

test('reduced motion disables workspace, spinner, and progress animations', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await waitForMission(page);
  await page.getByRole('button', { name: 'Run full pipeline' }).click();
  await expect(page.locator('.ops-header-progress')).toBeVisible();
  const animationNames = await page.evaluate(() => {
    const selectors = ['.ops-workspace', '.ops-icon.spin', '.ops-header-progress'];
    return selectors.map((selector) => {
      if (selector === '.ops-header-progress') {
        return getComputedStyle(document.querySelector(selector), '::after').animationName;
      }
      const element = document.querySelector(selector);
      return element ? getComputedStyle(element).animationName : 'none';
    });
  });
  expect(animationNames.every((name) => name === 'none')).toBe(true);
  await expect(page.locator('.ops-app')).toHaveAttribute('aria-busy', 'false', { timeout: 45_000 });
});
