/**
 * ui.js
 * ──────
 * Pure DOM helpers — zero networking logic.
 * Every function here only touches the DOM. No fetch, no WebSocket.
 */

// ── Language display names ───────────────────────────────────────────────────
const LANG_NAMES = {
  'hi-IN': 'हिन्दी (Hindi)',
  'ta-IN': 'தமிழ் (Tamil)',
  'te-IN': 'తెలుగు (Telugu)',
  'bn-IN': 'বাংলা (Bengali)',
  'mr-IN': 'मराठी (Marathi)',
  'gu-IN': 'ગુજરાતી (Gujarati)',
  'kn-IN': 'ಕನ್ನಡ (Kannada)',
  'ml-IN': 'മലയാളം (Malayalam)',
  'pa-IN': 'ਪੰਜਾਬੀ (Punjabi)',
  'od-IN': 'ଓଡ଼ିଆ (Odia)',
  'en-IN': 'English (Indian)',
};

/**
 * Add a transcript/translation bubble to the conversation area.
 * @param {'a'|'b'} speaker
 * @param {string}  text
 * @param {string}  lang - BCP-47 code
 */
export function addBubble(speaker, text, lang) {
  const container = document.getElementById(`transcript-${speaker}`);
  if (!container) return;

  // Remove the empty-state placeholder if present
  const empty = container.querySelector('.empty-state');
  if (empty) empty.remove();

  const bubble = document.createElement('div');
  bubble.className = speaker === 'a' ? 'bubble-a' : 'bubble-b';

  const langLabel = LANG_NAMES[lang] || lang || '';
  bubble.innerHTML = `
    <p class="text-white/90 break-words">${escapeHtml(text)}</p>
    ${langLabel ? `<span class="block mt-1 text-[10px] text-white/40">${escapeHtml(langLabel)}</span>` : ''}
  `;

  container.appendChild(bubble);
  // Auto-scroll to the latest bubble
  container.scrollTop = container.scrollHeight;
}

/**
 * Show/hide the processing spinner.
 * @param {'a'|'b'} speaker
 * @param {boolean} visible
 */
export function setSpinner(speaker, visible) {
  const el = document.getElementById(`spinner-${speaker}`);
  if (el) el.classList.toggle('hidden', !visible);
}

/**
 * Set the label text below the record button.
 * @param {'a'|'b'} speaker
 * @param {string}  text
 */
export function setLabel(speaker, text) {
  const el = document.getElementById(`label-${speaker}`);
  if (el) el.textContent = text;
}

/**
 * Update the live transcript preview pill.
 * @param {'a'|'b'} speaker
 * @param {string}  text
 */
export function setLiveTranscript(speaker, text) {
  const el = document.getElementById(`live-${speaker}`);
  if (!el) return;
  el.textContent = text;
  if (text) {
    el.className = 'live-pill-active';
  } else {
    el.className = 'live-pill';
    el.textContent = '';
  }
}

/**
 * Clear the live transcript.
 * @param {'a'|'b'} speaker
 */
export function clearLiveTranscript(speaker) {
  setLiveTranscript(speaker, '');
}

/**
 * Set the detected language label in the panel header.
 * @param {'a'|'b'} speaker
 * @param {string}  langCode - BCP-47
 */
export function setLanguageLabel(speaker, langCode) {
  const el = document.getElementById(`lang-${speaker}`);
  if (el) el.textContent = LANG_NAMES[langCode] || langCode || 'Auto-detecting…';
}

/**
 * Show a top-bar error that auto-dismisses after 5 seconds.
 * @param {string} message
 */
export function showError(message) {
  // Remove existing error bar if any
  const existing = document.getElementById('error-bar');
  if (existing) existing.remove();

  const bar = document.createElement('div');
  bar.id        = 'error-bar';
  bar.className = 'error-bar';
  bar.textContent = message;
  document.body.appendChild(bar);

  setTimeout(() => {
    if (bar.parentNode) {
      bar.style.opacity = '0';
      bar.style.transition = 'opacity 0.3s ease-out';
      setTimeout(() => bar.remove(), 300);
    }
  }, 5000);
}

/**
 * Update the record button visual state.
 * @param {'a'|'b'} speaker
 * @param {boolean} recording
 */
export function setRecordButton(speaker, recording) {
  const btn = document.getElementById(`btn-${speaker}`);
  if (!btn) return;

  const ping = document.getElementById(`ping-${speaker}`);

  if (recording) {
    btn.className = speaker === 'a' ? 'btn-record-active-a' : 'btn-record-active-b';
    btn.innerHTML = `
      <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
        <rect x="6" y="6" width="12" height="12" rx="2"/>
      </svg>
    `;
    if (ping) ping.classList.remove('hidden');
  } else {
    btn.className = 'btn-record-idle';
    btn.innerHTML = `
      <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
      </svg>
    `;
    if (ping) ping.classList.add('hidden');
  }
}

/**
 * Show the session badge.
 * @param {string} sessionId
 */
export function showSessionBadge(sessionId) {
  const badge = document.getElementById('session-badge');
  if (badge) {
    badge.textContent = `Session: ${sessionId.slice(0, 8)}…`;
    badge.classList.remove('hidden');
  }
}

/**
 * Show the audio streaming indicator.
 * @param {'a'|'b'} speaker
 * @param {boolean} playing
 */
export function setAudioPlaying(speaker, playing) {
  const el = document.getElementById(`audio-indicator-${speaker}`);
  if (el) el.classList.toggle('hidden', !playing);
}

// ── Internal helpers ────────────────────────────────────────────────────────

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
