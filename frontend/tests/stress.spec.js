// @ts-check
import { test, expect } from '@playwright/test';
import {
  navigateToApp,
  stubReset,
  waitForTurnState,
  getTestHooks,
  doFullTurn,
  stubHistory,
} from './helpers.js';

/**
 * stress.spec.js
 * ──────────────
 * Stage 3 — B20: Stress and leak detection.
 */

const TURN_COUNT = parseInt(process.env.VARTA_STRESS_TURNS ?? '20', 10);

test.describe('Stage 3 — Stress & Leak Detection', () => {

  test.beforeEach(async ({ page }) => {
    await stubReset(page);
  });

  test(`B20 — ${TURN_COUNT} sequential turns with no leaked state`, async ({ page }) => {
    test.setTimeout(TURN_COUNT * 15_000);

    await navigateToApp(page);

    const turnResults = [];
    const speakers = ['a', 'b'];

    for (let i = 0; i < TURN_COUNT; i++) {
      const speaker = speakers[i % 2];

      const beforeHooks = await getTestHooks(page, speaker);
      expect(beforeHooks.turnState).toBe('IDLE');

      const result = await doFullTurn(page, speaker, { timeout: 12_000 });

      turnResults.push({
        index: i,
        speaker,
        turnState: result.turnState,
        turnId: result.turnId,
      });

      expect(result.turnState).toBe('IDLE');
      expect(result.turnId).toBeNull();

      const activeSpeaker = await page.evaluate(() =>
        window.__vartaTestHooks.getActiveSpeaker()
      );
      expect(activeSpeaker).toBeNull();
    }

    // Post-run assertions
    const allIdle = turnResults.every(r => r.turnState === 'IDLE');
    expect(allIdle).toBe(true);

    const allCleared = turnResults.every(r => r.turnId === null);
    expect(allCleared).toBe(true);

    const history = await stubHistory(page);
    const turnStarts = history.events.filter(e => e.event === 'turn_start_received');
    expect(turnStarts.length).toBe(TURN_COUNT);

    const audioEnds = history.events.filter(e => e.event === 'audio_end_sent');
    expect(audioEnds.length).toBe(TURN_COUNT);

    for (const sp of ['a', 'b']) {
      const finalHooks = await getTestHooks(page, sp);
      expect(finalHooks.turnState).toBe('IDLE');
    }

    console.log(`[Stress] ✅ ${TURN_COUNT} turns completed cleanly`);
  });
});
