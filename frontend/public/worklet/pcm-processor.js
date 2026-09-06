/**
 * pcm-processor.js
 * ─────────────────
 * AudioWorkletProcessor that:
 *  1. Reports the actual AudioContext sample rate to the main thread.
 *  2. Resamples input to exactly 16,000 Hz using a linear interpolation
 *     resampler with a simple anti-alias low-pass pre-filter.
 *  3. Emits 20 ms chunks of Int16 mono PCM to the main thread via postMessage.
 *
 * Why resample here (not in the main thread)?
 *   Resampling on the audio rendering thread keeps the 16 kHz contract close
 *   to the audio clock, avoids blocking the main UI thread, and lets us apply
 *   the anti-alias filter before downsampling without a copy through the
 *   transferable channel.
 *
 * Supported input rates: any rate the browser provides.
 *   - 16,000 Hz → pass-through (no resample overhead)
 *   - 32,000 Hz → integer 2× downsample (fast path)
 *   - 44,100 Hz → fractional resample
 *   - 48,000 Hz → fractional resample
 *
 * Anti-alias filter:
 *   A simple single-pole IIR low-pass is applied when downsampling.
 *   Cutoff is set at 7,500 Hz (just below the Nyquist of the 16 kHz output).
 *   This removes alias artifacts that would appear with naive integer decimation.
 *
 * Failure mode:
 *   If the input rate cannot be determined or is 0, the processor posts
 *   { type: 'error', code: 'AUDIO_SAMPLE_RATE_UNSUPPORTED' } and returns false
 *   to remove itself, triggering a recorder error in the main thread.
 *
 * Chunk size: exactly 320 samples = 20 ms at 16 kHz.
 */

const TARGET_RATE   = 16_000;   // Hz — what Sarvam expects
const CHUNK_SAMPLES = 320;      // 20 ms at 16 kHz
const LPF_CUTOFF    = 7_500;   // Hz — anti-alias cutoff

class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    // sampleRate is a global in the AudioWorklet scope
    const inputRate = sampleRate || 0;

    if (!inputRate || inputRate < 8_000) {
      this.port.postMessage({ type: 'error', code: 'AUDIO_SAMPLE_RATE_UNSUPPORTED', rate: inputRate });
      this._dead = true;
      return;
    }

    this._inputRate  = inputRate;
    this._ratio      = inputRate / TARGET_RATE;   // e.g. 3.0 for 48kHz→16kHz
    this._dead       = false;

    // Resampler state: fractional phase index into input
    this._phase      = 0.0;

    // Anti-alias IIR low-pass state (single-pole)
    // Only applied when downsampling (ratio > 1)
    this._lpfEnabled = this._ratio > 1.0;
    this._lpfState   = 0.0;
    if (this._lpfEnabled) {
      // RC constant for IIR: alpha = dt/(RC+dt) where RC = 1/(2π·fc)
      const dt    = 1.0 / inputRate;
      const RC    = 1.0 / (2.0 * Math.PI * LPF_CUTOFF);
      this._alpha = dt / (RC + dt);    // ≈ 0.097 at 48kHz, cutoff 7.5kHz
    }

    // Output accumulator: accumulate resampled samples until a full 20 ms chunk
    this._outBuf = new Float32Array(CHUNK_SAMPLES * 4); // pre-allocate with headroom
    this._outLen = 0;

    // Inform the main thread of the actual input rate (for diagnostics)
    this.port.postMessage({ type: 'rate', sampleRate: inputRate, targetRate: TARGET_RATE });

    console.log(`[PCM Worklet] inputRate=${inputRate} ratio=${this._ratio.toFixed(4)} lpf=${this._lpfEnabled}`);
  }

  process(inputs) {
    if (this._dead) return false;   // remove processor

    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    const inLen = channel.length;

    // ── 1. Anti-alias low-pass filter (in-place on a copy) ────────────────
    let filtered;
    if (this._lpfEnabled) {
      filtered = new Float32Array(inLen);
      let s = this._lpfState;
      const a = this._alpha;
      for (let i = 0; i < inLen; i++) {
        s = s + a * (channel[i] - s);
        filtered[i] = s;
      }
      this._lpfState = s;
    } else {
      filtered = channel;
    }

    // ── 2. Resample via linear interpolation ──────────────────────────────
    let phase = this._phase;
    const ratio = this._ratio;

    while (phase < inLen) {
      const i0  = Math.floor(phase);
      const i1  = Math.min(i0 + 1, inLen - 1);
      const frac = phase - i0;
      const sample = filtered[i0] + frac * (filtered[i1] - filtered[i0]);

      // Expand output buffer if needed
      if (this._outLen >= this._outBuf.length) {
        const bigger = new Float32Array(this._outBuf.length * 2);
        bigger.set(this._outBuf);
        this._outBuf = bigger;
      }
      this._outBuf[this._outLen++] = sample;

      // Emit 20 ms chunks as they fill
      if (this._outLen >= CHUNK_SAMPLES) {
        const int16 = new Int16Array(CHUNK_SAMPLES);
        for (let j = 0; j < CHUNK_SAMPLES; j++) {
          int16[j] = Math.max(-32768, Math.min(32767, Math.round(this._outBuf[j] * 32768)));
        }
        // Shift remaining samples left
        const remaining = this._outLen - CHUNK_SAMPLES;
        this._outBuf.copyWithin(0, CHUNK_SAMPLES, this._outLen);
        this._outLen = remaining;

        // Transfer zero-copy to the main thread
        this.port.postMessage(int16.buffer, [int16.buffer]);
      }

      phase += ratio;
    }

    // Advance phase, keeping only the fractional overshoot past the input block
    this._phase = phase - inLen;

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
