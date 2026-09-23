// @ts-check
import { test, expect } from '@playwright/test';
import {
  navigateToApp,
  stubReset,
  stubHistory,
  waitForTurnState,
  getTestHooks,
  doFullTurn,
} from './helpers.js';

/**
 * sarvam-contract.spec.js
 * ───────────────────────
 * B21-B26: Sarvam provider contract assertions.
 */

test.describe('Sarvam Contract Validation', () => {

  test.beforeEach(async ({ page }) => {
    await stubReset(page);
    await navigateToApp(page);
  });

  test('B21 — server_ready sent before any turn interaction', async ({ page }) => {
    await page.evaluate(() => window.toggleRecord('a'));
    await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });

    const history = await stubHistory(page);
    const events = history.events;

    const serverReady = events.find(e => e.event === 'server_ready_sent');
    expect(serverReady).toBeTruthy();

    const readyIdx = events.indexOf(serverReady);
    const turnStart = events.find(e => e.event === 'turn_start_received');
    if (turnStart) {
      const startIdx = events.indexOf(turnStart);
      expect(readyIdx).toBeLessThan(startIdx);
    }

    await page.evaluate(() => window.toggleRecord('a'));
    await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });
  });


  test('B22 — Three turns follow correct event ordering', async ({ page }) => {
    for (let i = 0; i < 3; i++) {
      await doFullTurn(page, 'a');
    }

    const history = await stubHistory(page);
    const events = history.events;

    const turnStarts = events.filter(e => e.event === 'turn_start_received');
    expect(turnStarts.length).toBe(3);

    const audioEnds = events.filter(e => e.event === 'audio_end_sent');
    expect(audioEnds.length).toBe(3);

    for (const ts of turnStarts) {
      const matchingEnd = audioEnds.find(ae => ae.turn_id === ts.turn_id);
      expect(matchingEnd).toBeTruthy();
    }

    const stopRecordings = events.filter(e => e.event === 'stop_recording_received');
    expect(stopRecordings.length).toBe(3);
    for (const sr of stopRecordings) {
      expect(sr.chunks_received).toBeGreaterThan(0);
    }
  });


  test('B26 — App loads from the built static bundle', async ({ page }) => {
    const response = await page.goto('/static/');
    expect(response?.ok()).toBeTruthy();

    const hasHooks = await page.evaluate(() => !!window.__vartaTestHooks);
    expect(hasHooks).toBe(true);

    const turnStateEnum = await page.evaluate(() =>
      window.__vartaTestHooks.TurnState
    );
    expect(turnStateEnum).toHaveProperty('IDLE');
    expect(turnStateEnum).toHaveProperty('RECORDING');
    expect(turnStateEnum).toHaveProperty('WAITING');
    expect(turnStateEnum).toHaveProperty('PLAYING');
  });
});
