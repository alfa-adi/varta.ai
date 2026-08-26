/**
 * player.js
 * ──────────
 * Gapless TTS audio player.
 *
 * Receives base64-encoded MP3 chunks from the server and queues them for
 * back-to-back playback with zero gaps between chunks.
 *
 * Uses HTML5 Audio elements for broad compatibility (Web Audio API
 * decodeAudioData can be flaky with MP3 streaming on some browsers).
 */

export class AudioPlayer {
  constructor() {
    /** @type {string[]} Queue of base64 data strings */
    this._queue     = [];
    /** @type {boolean} */
    this._isPlaying = false;
    /** @type {HTMLAudioElement|null} */
    this._current   = null;

    /** @type {function(): void} Called when all audio finishes */
    this.onFinished = null;

    /** @type {function(boolean): void} Called when playing state changes */
    this.onStateChange = null;
  }

  /**
   * Add a base64-encoded MP3 chunk to the queue.
   * Starts playback immediately if not already playing.
   * @param {string} base64data - base64-encoded MP3 bytes
   */
  enqueue(base64data) {
    this._queue.push(base64data);
    if (!this._isPlaying) {
      this._playNext();
    }
  }

  /**
   * Flush the queue — if something is still queued, keep playing.
   * Called on audio_end to ensure the last chunk gets played.
   */
  flush() {
    // If not currently playing but items remain, kick off playback
    if (!this._isPlaying && this._queue.length > 0) {
      this._playNext();
    }
  }

  /**
   * Stop all playback and clear the queue.
   */
  clear() {
    this._queue = [];
    if (this._current) {
      this._current.pause();
      this._current.src = '';
      this._current = null;
    }
    this._isPlaying = false;
    if (this.onStateChange) this.onStateChange(false);
  }

  /** @private */
  _playNext() {
    if (this._queue.length === 0) {
      this._isPlaying = false;
      if (this.onStateChange) this.onStateChange(false);
      if (this.onFinished) this.onFinished();
      return;
    }

    this._isPlaying = true;
    if (this.onStateChange) this.onStateChange(true);

    const data    = this._queue.shift();
    const dataUrl = `data:audio/mp3;base64,${data}`;
    const audio   = new Audio(dataUrl);

    this._current = audio;

    audio.onended = () => {
      this._current = null;
      this._playNext();
    };

    audio.onerror = (e) => {
      console.error('[Player] Audio playback error:', e);
      this._current = null;
      this._playNext(); // skip broken chunk, play next
    };

    audio.play().catch((e) => {
      console.error('[Player] Play failed:', e);
      this._current = null;
      this._playNext();
    });
  }
}
