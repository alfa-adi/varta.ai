/**
 * session.js
 * ──────────
 * Creates and manages translation sessions via the REST API.
 *
 * Endpoint: POST /session/create
 * Body:     multipart/form-data { lang_a, lang_b }
 * Returns:  { session_id, lang_a, lang_b }
 */

/**
 * Create a new bilingual conversation session.
 * @param {string} langA - BCP-47 code for Speaker A (e.g. "hi-IN"), or empty for auto-detect.
 * @param {string} langB - BCP-47 code for Speaker B, or empty for auto-detect.
 * @returns {Promise<{session_id: string, lang_a: string|null, lang_b: string|null}>}
 */
export async function createSession(langA = '', langB = '') {
  const form = new FormData();
  form.append('lang_a', langA);
  form.append('lang_b', langB);

  const res = await fetch('/session/create', { method: 'POST', body: form });

  if (!res.ok) {
    throw new Error(`Session creation failed (HTTP ${res.status})`);
  }

  return await res.json();
}
