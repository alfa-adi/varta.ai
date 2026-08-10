/**
 * pcm-processor.js
 * ─────────────────
 * AudioWorkletProcessor that converts browser Float32 samples to 16-bit
 * signed integer PCM (pcm_s16le) and sends 20ms chunks to the main thread.
 *
 * Runs on the audio rendering thread (off the main JS thread) for zero-
 * latency capture. Each postMessage call sends a transferable ArrayBuffer
 * of Int16 samples — the main thread forwards these directly over WebSocket
 * to the server, which relays them to Sarvam saaras:v3-realtime.
 *
 * Audio spec: 16kHz, mono, 16-bit signed little-endian (pcm_s16le)
 * Chunk size: 320 samples = 20ms at 16kHz (one WebSocket frame per chunk)
 */

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Buffer to accumulate samples until we have a full 20ms chunk (320 samples)
    this._buf = [];
    this._chunkSize = 320; // 16000 Hz * 0.020s = 320 samples per 20ms
  }

  process(inputs) {
    // inputs[0] = first input, inputs[0][0] = first (mono) channel
    const channel = inputs[0]?.[0];
    if (!channel) return true; // keep processor alive even if no input yet

    // Accumulate Float32 samples into buffer
    for (let i = 0; i < channel.length; i++) {
      this._buf.push(channel[i]);
    }

    // Flush complete 20ms chunks
    while (this._buf.length >= this._chunkSize) {
      const chunk = this._buf.splice(0, this._chunkSize);
      const int16 = new Int16Array(this._chunkSize);

      for (let i = 0; i < this._chunkSize; i++) {
        // Clamp Float32 (-1.0 to 1.0) to Int16 range (-32768 to 32767)
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(chunk[i] * 32768)));
      }

      // Transfer ownership of the underlying ArrayBuffer — zero-copy to main thread
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true; // return true to keep processor alive
  }
}

registerProcessor('pcm-processor', PCMProcessor);
