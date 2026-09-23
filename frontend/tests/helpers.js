// @ts-check
/**
 * tests/helpers.js
 * ────────────────
 * Shared helpers for all Varta Playwright test suites.
 */

import { expect } from '@playwright/test';

const STUB_BASE = 'http://127.0.0.1:8000';

/**
 * Inject a WebSocket frame to an active connection on the stub server.
 * @param {import('@playwright/test').Page} page
 * @param {string} sessionId
 * @param {'a'|'b'} speaker
 * @param {object} message
 */
export async function stubInject(page, sessionId, speaker, message) {
  const res = await page.request.post(`${STUB_BASE}/_stub/inject`, {
    data: { session_id: sessionId, speaker, message },
  });
  expect(res.ok()).toBeTruthy();
}

/**
 * Configure the stub server behavior.
 * @param {import('@playwright/test').Page} page
 * @param {object} config
 */
export async function stubConfigure(page, config) {
  const res = await page.request.post(`${STUB_BASE}/_stub/configure`, {
    data: config,
  });
  expect(res.ok()).toBeTruthy();
  return res.json();
}

/**
 * Reset stub to default happy-path behavior.
 * @param {import('@playwright/test').Page} page
 */
export async function stubReset(page) {
  const res = await page.request.post(`${STUB_BASE}/_stub/reset`);
  expect(res.ok()).toBeTruthy();
}

/**
 * Fetch the upstream event history from the stub.
 * @param {import('@playwright/test').Page} page
 * @returns {Promise<{events: Array<object>}>}
 */
export async function stubHistory(page) {
  const res = await page.request.get(`${STUB_BASE}/_stub/history`);
  expect(res.ok()).toBeTruthy();
  return res.json();
}

/**
 * Navigate to the app and wait for it to be ready.
 * @param {import('@playwright/test').Page} page
 */
export async function navigateToApp(page) {
  await page.goto('/static/');
  // Wait for Varta app to boot — the test hooks are exposed on window
  await page.waitForFunction(() => !!window.__vartaTestHooks, null, { timeout: 10_000 });
}

/**
 * Wait for a speaker to reach a specific TurnState.
 * @param {import('@playwright/test').Page} page
 * @param {'a'|'b'} speaker
 * @param {string} state
 * @param {{ timeout?: number }} options
 */
export async function waitForTurnState(page, speaker, state, options = {}) {
  const timeout = options.timeout ?? 20_000;
  await page.waitForFunction(
    ([sp, st]) => {
      const hooks = window.__vartaTestHooks;
      return hooks && hooks.getTurnState(sp) === st;
    },
    [speaker, state],
    { timeout }
  );
}

/**
 * Get a complete snapshot of test hook data for a speaker.
 * @param {import('@playwright/test').Page} page
 * @param {'a'|'b'} speaker
 */
export async function getTestHooks(page, speaker) {
  return page.evaluate((sp) => {
    const h = window.__vartaTestHooks;
    if (!h) return null;
    return {
      turnState:     h.getTurnState(sp),
      turnId:        h.getTurnId(sp),
      wsState:       h.getWSState(sp),
      player:        h.getPlayerCounters(sp),
      recorder:      h.getRecorderStats(sp),
      sessionId:     h.getSessionId(),
      activeSpeaker: h.getActiveSpeaker(),
    };
  }, speaker);
}

/**
 * Execute a complete turn: click record → wait for RECORDING → click stop → wait for IDLE.
 * @param {import('@playwright/test').Page} page
 * @param {'a'|'b'} speaker
 * @param {{ timeout?: number }} options
 */
export async function doFullTurn(page, speaker, options = {}) {
  const timeout = options.timeout ?? 30_000;

  // Start recording
  await page.click(`#btn-${speaker}`);
  await waitForTurnState(page, speaker, 'RECORDING', { timeout: 10_000 });

  // Let some audio chunks flow (fake mic generates audio automatically)
  await page.waitForTimeout(1500);

  // Stop recording
  await page.click(`#btn-${speaker}`);

  // Wait for the turn to complete → IDLE
  await waitForTurnState(page, speaker, 'IDLE', { timeout });

  // Return final state
  return getTestHooks(page, speaker);
}
