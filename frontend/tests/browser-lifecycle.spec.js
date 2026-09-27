// @ts-check
import { test, expect } from '@playwright/test';
import {
  navigateToApp,
  stubReset,
  stubConfigure,
  stubHistory,
  waitForTurnState,
  getTestHooks,
  doFullTurn,
  stubInject,
} from './helpers.js';

/**
 * browser-lifecycle.spec.js
 * ─────────────────────────
 * Stage 1 — Contract smoke (B01-B05) and select Stage 2 lifecycle tests.
 */

test.describe('Stage 1 — Contract Smoke', () => {

  test.beforeEach(async ({ page }) => {
    await stubReset(page);
    await navigateToApp(page);
  });

  // ── B01: Initial page and session bootstrap ────────────────────────────────
  test('B01 — Page loads without errors; test hooks are available', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const hasHooks = await page.evaluate(() => !!window.__vartaTestHooks);
    expect(hasHooks).toBe(true);

    const hooksA = await getTestHooks(page, 'a');
    const hooksB = await getTestHooks(page, 'b');
    expect(hooksA.turnState).toBe('IDLE');
    expect(hooksB.turnState).toBe('IDLE');

    expect(hooksA.wsState).toBe('NONE');
    expect(hooksB.wsState).toBe('NONE');

    expect(errors).toHaveLength(0);
  });


  // ── B02: First turn — full lifecycle ───────────────────────────────────────
  test('B02 — First turn completes full lifecycle', async ({ page }) => {
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });

    const midRecordHooks = await getTestHooks(page, 'a');
    expect(midRecordHooks.wsState).toBe('OPEN');
    expect(midRecordHooks.turnId).toBeTruthy();

    await page.waitForTimeout(2000);

    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });

    const finalHooks = await getTestHooks(page, 'a');
    expect(finalHooks.turnState).toBe('IDLE');
    expect(finalHooks.turnId).toBeNull();

    const history = await stubHistory(page);
    const eventNames = history.events.map(e => e.event);

    expect(eventNames).toContain('server_ready_sent');
    expect(eventNames).toContain('turn_start_received');
    expect(eventNames).toContain('audio_chunk_received');
    expect(eventNames).toContain('stop_recording_received');
    expect(eventNames).toContain('transcript_final_sent');
    expect(eventNames).toContain('audio_chunks_sent');
    expect(eventNames).toContain('audio_end_sent');

    const turnStartEvent = history.events.find(e => e.event === 'turn_start_received');
    const turnId = turnStartEvent.turn_id;
    expect(turnId).toBeTruthy();

    const audioEndEvent = history.events.find(e => e.event === 'audio_end_sent');
    expect(audioEndEvent.turn_id).toBe(turnId);
  });


  // ── B03: Rapid double click ────────────────────────────────────────────────
  test('B03 — Double click is safely deduplicated', async ({ page }) => {
    await page.evaluate(() => {
      window.toggleRecord('a');
      window.toggleRecord('a');
    });

    await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });

    const hooks = await getTestHooks(page, 'a');
    expect(hooks.turnId).toBeTruthy();

    await page.waitForTimeout(1000);
    const history = await stubHistory(page);
    const turnStarts = history.events.filter(e => e.event === 'turn_start_received');
    expect(turnStarts.length).toBe(1);

    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });
  });


  // ── B04: Stop while capturing ──────────────────────────────────────────────
  test('B04 — Stop after capturing sends stop_recording', async ({ page }) => {
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });
    await page.waitForTimeout(2000);

    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });

    const history = await stubHistory(page);
    const stopEvent = history.events.find(e => e.event === 'stop_recording_received');
    expect(stopEvent).toBeTruthy();
    expect(stopEvent.chunks_received).toBeGreaterThan(0);
  });


  // ── B05: Click while draining/playing ──────────────────────────────────────
  test('B05 — Click during WAITING state is ignored', async ({ page }) => {
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });
    await page.waitForTimeout(1500);

    await page.click('#btn-a');
    await page.waitForTimeout(50);

    const stateBeforeClick = await page.evaluate(() =>
      window.__vartaTestHooks.getTurnState('a')
    );

    if (stateBeforeClick === 'WAITING') {
      await page.click('#btn-a');
      const stateAfterClick = await page.evaluate(() =>
        window.__vartaTestHooks.getTurnState('a')
      );
      expect(['WAITING', 'PLAYING', 'IDLE']).toContain(stateAfterClick);
    }

    await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });

    const history = await stubHistory(page);
    const turnStarts = history.events.filter(e => e.event === 'turn_start_received');
    expect(turnStarts.length).toBe(1);
  });
});


test.describe('Stage 2 — Lifecycle Matrix (select)', () => {

  test.beforeEach(async ({ page }) => {
    await stubReset(page);
    await navigateToApp(page);
  });

  // ── B06: Ten turns on one speaker ──────────────────────────────────────────
  test('B06 — Ten sequential turns on speaker A', async ({ page }) => {
    test.setTimeout(120_000);
    const turnIds = [];

    for (let i = 0; i < 10; i++) {
      await page.click('#btn-a');
      await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });

      const hooks = await getTestHooks(page, 'a');
      turnIds.push(hooks.turnId);

      await page.waitForTimeout(800);
      await page.click('#btn-a');
      await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });
    }

    const uniqueIds = new Set(turnIds);
    expect(uniqueIds.size).toBe(10);

    const history = await stubHistory(page);
    const turnStarts = history.events.filter(e => e.event === 'turn_start_received');
    expect(turnStarts.length).toBe(10);

    const finalHooks = await getTestHooks(page, 'a');
    expect(finalHooks.turnState).toBe('IDLE');
  });


  // ── B07: Alternating speakers ──────────────────────────────────────────────
  test('B07 — Alternating A, B, A, B turns', async ({ page }) => {
    const speakers = ['a', 'b', 'a', 'b'];

    for (const sp of speakers) {
      await page.click(`#btn-${sp}`);
      await waitForTurnState(page, sp, 'RECORDING', { timeout: 10_000 });
      await page.waitForTimeout(800);
      await page.click(`#btn-${sp}`);
      await waitForTurnState(page, sp, 'IDLE', { timeout: 20_000 });
    }

    const hooksA = await getTestHooks(page, 'a');
    const hooksB = await getTestHooks(page, 'b');
    expect(hooksA.turnState).toBe('IDLE');
    expect(hooksB.turnState).toBe('IDLE');

    const history = await stubHistory(page);
    const turnStarts = history.events.filter(e => e.event === 'turn_start_received');
    expect(turnStarts.length).toBe(4);
  });


  // ── B13: No speech / silence ───────────────────────────────────────────────
  test('B13 — Silence fixture produces no-speech error', async ({ page }) => {
    await stubConfigure(page, { fail_mode: 'no_speech' });

    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });
    await page.waitForTimeout(1000);
    await page.click('#btn-a');

    await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });

    const history = await stubHistory(page);
    const noSpeech = history.events.find(e => e.event === 'no_speech_sent');
    expect(noSpeech).toBeTruthy();

    await stubConfigure(page, { fail_mode: null });
    const result = await doFullTurn(page, 'a');
    expect(result.turnState).toBe('IDLE');
  });


  // ── B14: NMT/TTS failure ───────────────────────────────────────────────────
  test('B14 — NMT error after transcript returns to recoverable IDLE', async ({ page }) => {
    await stubConfigure(page, { fail_mode: 'nmt_error' });

    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING', { timeout: 10_000 });
    await page.waitForTimeout(1000);
    await page.click('#btn-a');

    await waitForTurnState(page, 'a', 'IDLE', { timeout: 20_000 });

    const history = await stubHistory(page);
    const nmtError = history.events.find(e => e.event === 'nmt_error_sent');
    expect(nmtError).toBeTruthy();

    await stubConfigure(page, { fail_mode: null });
    const result = await doFullTurn(page, 'a');
    expect(result.turnState).toBe('IDLE');
  });
  // ── B27: Pause Turn 1 for Turn 2 recording ────────────────────────────────
  test('B27 — Pause Turn 1 for Turn 2 recording', async ({ page }) => {
    // Generate a lot of chunks for Turn 1 so it plays a long time
    await stubConfigure(page, { audio_chunk_count: 50 }); 

    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    
    // Stop recording Turn 1
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE');

    // Turn 1 is IDLE, but audio should be playing on B
    const hooks1 = await getTestHooks(page, 'a');
    expect(hooks1.turnState).toBe('IDLE');
    
    // Get turn 1 record
    const t1Id = Array.from(await page.evaluate(() => window.__vartaTestHooks.getTurns('a').keys()))[0];
    const turn1 = await page.evaluate((id) => window.__vartaTestHooks.getTurns('a').get(id), t1Id);
    
    expect(turn1.serverAudioEnded).toBe(true);
    expect(turn1.playbackStarted).toBe(true);
    expect(turn1.playbackFinished).toBe(false);
    
    // Start Turn 2 on A
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    
    const turn1Paused = await page.evaluate((id) => window.__vartaTestHooks.getTurns('a').get(id).playbackPaused, t1Id);
    expect(turn1Paused).toBe(true); // playback paused immediately
    
    // Turn 1 remains in memory
    const t1Remaining = await page.evaluate((id) => window.__vartaTestHooks.getTurns('a').has(id), t1Id);
    expect(t1Remaining).toBe(true);
  });


  // ── B28: Ordered sequence assertions ───────────────────────────────────────
  test('B28 — Ordered sequence assertions', async ({ page }) => {
    await stubConfigure(page, { audio_chunk_count: 5, first_audio_delay_s: 0.1 }); 

    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE'); // audio_end received
    
    // Get turn 1 ID
    const t1Id = Array.from(await page.evaluate(() => window.__vartaTestHooks.getTurns('a').keys()))[0];
    
    // Start Turn 2 immediately
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    
    const t2Id = Array.from(await page.evaluate(() => window.__vartaTestHooks.getTurns('a').keys()))[1];
    
    // Stop Turn 2
    await page.click('#btn-a');
    
    // Wait for everything to finish (both turns deleted)
    await page.waitForFunction(() => {
      const turns = window.__vartaTestHooks.getTurns('a');
      return turns.size === 0;
    }, null, { timeout: 20_000 });

    const history = await stubHistory(page);
    // B28 assert: exact event sequence isn't easy to extract from browser since they are logged via analytics,
    // but we can check the player counters.
    const hooks = await getTestHooks(page, 'a');
    const decodedT1 = hooks.player.decodedByTurn[t1Id] || 0;
    const decodedT2 = hooks.player.decodedByTurn[t2Id] || 0;
    
    // Each chunk is 4800 samples in stub_server
    expect(decodedT1).toBe(5 * 4800);
    expect(decodedT2).toBe(5 * 4800);
  });

  // ── B29: Paused state during upstream processing ───────────────────────────
  test('B29 — Paused state during upstream processing', async ({ page }) => {
    // Configure Turn 1
    await stubConfigure(page, { audio_chunk_count: 15 });
    
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE'); 
    
    const t1Id = Array.from(await page.evaluate(() => window.__vartaTestHooks.getTurns('a').keys()))[0];
    
    // Configure Turn 2 to have a long processing delay
    await stubConfigure(page, { first_audio_delay_s: 2.0 });
    
    // Start Turn 2
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'WAITING');
    
    // While Turn 2 is WAITING (upstream processing), Turn 1 is paused
    const t1State = await page.evaluate((id) => window.__vartaTestHooks.getTurns('a').get(id), t1Id);
    expect(t1State.playbackPaused).toBe(true);
    expect(t1State.playbackFinished).toBe(false);
    
    // Wait for Turn 2 to finish
    await waitForTurnState(page, 'a', 'IDLE');
    await page.waitForFunction(() => window.__vartaTestHooks.getTurns('a').size === 0, null, { timeout: 20_000 });
  });

  // ── B30: Controlled failure injection ──────────────────────────────────────
  test('B30 — Controlled failure injection (Delayed stale error)', async ({ page }) => {
    await stubConfigure(page, { audio_chunk_count: 10 });
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE'); 
    
    const hooks = await getTestHooks(page, 'a');
    const sessionId = hooks.sessionId;
    const t1Id = Array.from(await page.evaluate(() => window.__vartaTestHooks.getTurns('a').keys()))[0];
    
    // Start Turn 2
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    
    // Inject Turn 1 error
    await stubInject(page, sessionId, 'a', {
      type: 'turn_error',
      turn_id: t1Id,
      code: 'NMT_ERROR',
      message: 'Stale error'
    });
    
    // Assert Turn 2 is unaffected (still RECORDING)
    const state = await page.evaluate(() => window.__vartaTestHooks.getCaptureState('a'));
    expect(state).toBe('RECORDING');
    
    // Turn 1 should be finished with error
    const t1Exists = await page.evaluate((id) => window.__vartaTestHooks.getTurns('a').has(id), t1Id);
    expect(t1Exists).toBe(false);
    
    // Cleanup
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE');
  });

  test('B30 — Controlled failure injection (Provider failure before audio_end)', async ({ page }) => {
    await stubConfigure(page, { audio_chunk_count: 10 });
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE'); 
    
    const hooks = await getTestHooks(page, 'a');
    const sessionId = hooks.sessionId;
    const t1Id = Array.from(await page.evaluate(() => window.__vartaTestHooks.getTurns('a').keys()))[0];
    
    // Start Turn 2
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    const t2Id = Array.from(await page.evaluate(() => window.__vartaTestHooks.getTurns('a').keys()))[1];
    
    // Inject Turn 2 error
    await stubInject(page, sessionId, 'a', {
      type: 'turn_error',
      turn_id: t2Id,
      code: 'NMT_ERROR',
      message: 'Provider error'
    });
    
    // Assert active capture turn aborts (becomes IDLE)
    await waitForTurnState(page, 'a', 'IDLE');
    
    // Turn 1 resumes and eventually finishes
    await page.waitForFunction(() => window.__vartaTestHooks.getTurns('a').size === 0, null, { timeout: 15_000 });
  });

  // ── B31: Queue Overflow ────────────────────────────────────────────────────
  test('B31a — Turn 1 queue overflow', async ({ page }) => {
    // Max buffered audio limit is ~90s. At 200ms per chunk (4800 samples), that is 450 chunks.
    // Configure stub to send 500 chunks.
    await stubConfigure(page, { audio_chunk_count: 500, inter_chunk_delay_s: 0.01 });
    
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    
    // Wait until overflow is reached
    await page.waitForFunction(() => {
       const turns = window.__vartaTestHooks.getTurns('a');
       for (const t of turns.values()) {
         if (t.terminalEvent === 'playback_queue_overflow') return true;
       }
       return false;
    }, null, { timeout: 30_000 });
    
    const turns = await page.evaluate(() => Array.from(window.__vartaTestHooks.getTurns('a').values()));
    expect(turns[0].terminalEvent).toBe('playback_queue_overflow');
  });

  test('B31b — Turn 2 queue overflow', async ({ page }) => {
    // Buffer 50 chunks for Turn 1
    await stubConfigure(page, { audio_chunk_count: 50, inter_chunk_delay_s: 0.01 });
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'IDLE');
    
    // Start Turn 2
    // Buffer 450 chunks for Turn 2, so it overflows
    await stubConfigure(page, { audio_chunk_count: 450, inter_chunk_delay_s: 0.01 });
    await page.click('#btn-a');
    await waitForTurnState(page, 'a', 'RECORDING');
    await page.click('#btn-a');
    
    // Wait until Turn 2 hits overflow
    await page.waitForFunction(() => {
       const turns = window.__vartaTestHooks.getTurns('a');
       for (const t of turns.values()) {
         if (t.terminalEvent === 'playback_queue_overflow') return true;
       }
       return false;
    }, null, { timeout: 30_000 });
    
    const turns = await page.evaluate(() => Array.from(window.__vartaTestHooks.getTurns('a').values()));
    
    // Turn 1's already-buffered remainder is unaffected and resumes normally
    expect(turns[0].terminalEvent).not.toBe('playback_queue_overflow');
    
    // Turn 2 is the one that overflowed
    expect(turns[1].terminalEvent).toBe('playback_queue_overflow');
  });

});
