"""
test/ws_reliability_test/reporter.py
──────────────────────────────────────
Generates all output artifacts from the 120-turn experiment:

  Local artifacts:
    results/run_YYYYMMDD_HHMMSS/
      summary.json          — ExperimentResult as JSON
      pipeline_events.jsonl — one JSON line per TurnResult (120 lines)
      failures.jsonl        — failed/partial turns only
      report.html           — human-readable HTML report

  MongoDB (new database "varta_ws_test" on same cluster):
    Collections:
      ws_test_sessions — one document per experiment run (session-level)
      ws_test_turns    — one document per TurnResult (turn-level, 120 docs per run)

MongoDB schema decision:
  - New database "varta_ws_test" is created automatically by MongoDB on first insert
  - Same cluster/connection string as production (MONGO_URL from .env)
  - Indexes created: ws_test_turns.run_id, ws_test_turns.conversation_id,
                     ws_test_turns.source_language, ws_test_turns.status
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import ExperimentResult, TurnStatus, FailureStage

# MongoDB database name for test results (separate from production)
TEST_DB_NAME = "varta_ws_test"


# ── Local artifact writer ─────────────────────────────────────────────────────

class LocalReporter:
    """Writes local JSON + JSONL + HTML artifacts."""

    def __init__(self, results_base_dir: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = results_base_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Reporter] Output directory: {self.run_dir}")

    def write_all(self, experiment: ExperimentResult) -> None:
        """Write all local artifacts."""
        self._write_summary(experiment)
        self._write_pipeline_events(experiment)
        self._write_failures(experiment)
        self._write_html_report(experiment)
        print(f"  [Reporter] All artifacts written to {self.run_dir}")

    def _write_summary(self, experiment: ExperimentResult) -> None:
        path = self.run_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)
        print(f"  [Reporter] summary.json written ({path.stat().st_size:,} bytes)")

    def _write_pipeline_events(self, experiment: ExperimentResult) -> None:
        path = self.run_dir / "pipeline_events.jsonl"
        turns = experiment.all_turns()
        with open(path, "w", encoding="utf-8") as f:
            for turn in turns:
                f.write(json.dumps(turn.to_dict(), default=str) + "\n")
        print(f"  [Reporter] pipeline_events.jsonl written ({len(turns)} turns)")

    def _write_failures(self, experiment: ExperimentResult) -> None:
        path = self.run_dir / "failures.jsonl"
        failures = [t for t in experiment.all_turns()
                    if t.status in (TurnStatus.FAILURE, TurnStatus.PARTIAL_SUCCESS)]
        with open(path, "w", encoding="utf-8") as f:
            for turn in failures:
                f.write(json.dumps(turn.to_dict(), default=str) + "\n")
        print(f"  [Reporter] failures.jsonl written ({len(failures)} non-clean turns)")

    def _write_html_report(self, experiment: ExperimentResult) -> None:
        path = self.run_dir / "report.html"
        html = _build_html_report(experiment)
        path.write_text(html, encoding="utf-8")
        print(f"  [Reporter] report.html written ({path.stat().st_size:,} bytes)")

    @property
    def run_directory(self) -> Path:
        return self.run_dir


# ── MongoDB writer ────────────────────────────────────────────────────────────

class MongoReporter:
    """
    Writes experiment results to MongoDB.

    Database: varta_ws_test  (new, separate from production)
    Collections:
      ws_test_sessions — session-level summary (one per run)
      ws_test_turns    — individual turn records (120 per run)
    """

    def __init__(self, mongo_url: str):
        self._mongo_url = mongo_url
        self._client    = None
        self._db        = None

    def connect(self) -> bool:
        """
        Attempt MongoDB connection. Returns True on success, False on failure.
        Never raises — MongoDB being down must not crash the test.
        """
        try:
            from pymongo import MongoClient, ASCENDING
            self._client = MongoClient(
                self._mongo_url,
                serverSelectionTimeoutMS=5000,
                appName="varta-ws-test",
            )
            self._client.server_info()   # fast fail if unreachable
            self._db = self._client[TEST_DB_NAME]

            # Ensure indexes exist (idempotent)
            turns_col = self._db["ws_test_turns"]
            turns_col.create_index([("run_id", ASCENDING)])
            turns_col.create_index([("conversation_id", ASCENDING)])
            turns_col.create_index([("source_language", ASCENDING)])
            turns_col.create_index([("status", ASCENDING)])
            turns_col.create_index([("turn_id", ASCENDING)], unique=True, sparse=True)

            sessions_col = self._db["ws_test_sessions"]
            sessions_col.create_index([("run_id", ASCENDING)], unique=True)

            print(f"  [MongoDB] Connected → database: '{TEST_DB_NAME}'")
            return True
        except Exception as exc:
            print(f"  [MongoDB] Connection failed ({exc}) — skipping MongoDB write")
            self._db = None
            return False

    def write_session(self, experiment: ExperimentResult) -> Optional[str]:
        """
        Insert session-level document into ws_test_sessions.
        Returns inserted document id as string, or None on failure.
        """
        if self._db is None:
            return None
        try:
            doc = experiment.to_dict()
            doc["_recorded_at"] = datetime.now(timezone.utc)
            # Remove nested turns from session doc — turns are in ws_test_turns
            doc.pop("conversations", None)
            result = self._db["ws_test_sessions"].insert_one(doc)
            print(f"  [MongoDB] Session document inserted: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as exc:
            print(f"  [MongoDB] Session write failed: {exc}")
            return None

    def write_turns(self, experiment: ExperimentResult) -> int:
        """
        Insert one document per TurnResult into ws_test_turns.
        Returns number of documents inserted.
        """
        if self._db is None:
            return 0

        turns = experiment.all_turns()
        docs = []
        for turn in turns:
            doc = turn.to_dict()
            doc["run_id"]       = experiment.run_id
            doc["run_label"]    = experiment.run_label
            doc["_recorded_at"] = datetime.now(timezone.utc)
            docs.append(doc)

        if not docs:
            return 0

        try:
            result = self._db["ws_test_turns"].insert_many(docs, ordered=False)
            n = len(result.inserted_ids)
            print(f"  [MongoDB] {n} turn documents inserted into ws_test_turns")
            return n
        except Exception as exc:
            print(f"  [MongoDB] Turn bulk insert failed: {exc}")
            # Try one-by-one as fallback
            inserted = 0
            for doc in docs:
                try:
                    self._db["ws_test_turns"].insert_one(doc)
                    inserted += 1
                except Exception:
                    pass
            print(f"  [MongoDB] Fallback: {inserted}/{len(docs)} turns inserted")
            return inserted

    def write_all(self, experiment: ExperimentResult) -> None:
        """Write session + all turns to MongoDB."""
        self.write_session(experiment)
        self.write_turns(experiment)

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass


# ── HTML report builder ───────────────────────────────────────────────────────

def _build_html_report(exp: ExperimentResult) -> str:
    """Build a self-contained HTML reliability and latency report."""

    turns = exp.all_turns()

    # ── Helpers ───────────────────────────────────────────────────────
    def status_badge(status: str) -> str:
        colors = {
            "CLEAN_SUCCESS":    "#22c55e",
            "PARTIAL_SUCCESS":  "#f59e0b",
            "FAILURE":          "#ef4444",
            "NOT_RUN":          "#6b7280",
        }
        color = colors.get(status, "#6b7280")
        label = status.replace("_", " ")
        return f'<span style="background:{color};color:#fff;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:700">{label}</span>'

    def fmt_ms(val) -> str:
        if val is None:
            return "—"
        return f"{int(val):,}ms"

    def pct_color(pct: float) -> str:
        if pct >= 90:   return "#22c55e"
        if pct >= 70:   return "#f59e0b"
        return "#ef4444"

    # ── Top-level summary table ───────────────────────────────────────
    run_start = datetime.fromtimestamp(exp.started_at).strftime("%Y-%m-%d %H:%M:%S") if exp.started_at else "—"
    duration_min = int((exp.finished_at - exp.started_at) / 60) if exp.finished_at and exp.started_at else 0

    summary_rows = [
        ("Total turns",        exp.total_turns, ""),
        ("Clean successes",    exp.clean_successes,   f"<span style='color:#22c55e'>✅</span>"),
        ("Partial successes",  exp.partial_successes, f"<span style='color:#f59e0b'>⚠️</span>"),
        ("Failures",           exp.failures,          f"<span style='color:#ef4444'>❌</span>"),
        ("Not run",            exp.not_run,            ""),
    ]

    summary_html = "".join(
        f"<tr><td>{icon} {label}</td><td><strong>{val}</strong></td></tr>"
        for label, val, icon in summary_rows
    )

    # ── Reliability summary ───────────────────────────────────────────
    cr = exp.clean_reliability_pct
    uv = exp.user_visible_success_pct
    ps = exp.pipeline_success_pct

    # ── Language table ────────────────────────────────────────────────
    lang_rows = ""
    for conv in exp.conversations:
        cr_c = conv.reliability_pct
        lang_rows += f"""
        <tr>
            <td><strong>{conv.language}</strong></td>
            <td>{conv.total_turns}</td>
            <td style="color:{pct_color(cr_c)};font-weight:700">{cr_c:.1f}%</td>
            <td style="color:{pct_color(conv.pipeline_success_pct)}">{conv.pipeline_success_pct:.1f}%</td>
            <td>{conv.clean_successes}</td>
            <td>{conv.partial_successes}</td>
            <td>{conv.failures}</td>
            <td>{fmt_ms(conv.median_e2e_latency_ms)}</td>
            <td>{fmt_ms(conv.median_asr_latency_ms)}</td>
            <td>{fmt_ms(conv.median_pipeline_latency_ms)}</td>
            <td>{conv.primary_failure_stage or '—'}</td>
        </tr>"""

    # ── Failure taxonomy table ────────────────────────────────────────
    fb = exp.failure_breakdown if hasattr(exp, 'failure_breakdown') else {}
    breakdown = exp.to_dict().get("failure_breakdown", {})
    failure_rows = "".join(
        f"<tr><td>{stage.replace('_',' ').title()}</td><td>{count}</td></tr>"
        for stage, count in breakdown.items() if count > 0
    )

    # ── Latency stats table ───────────────────────────────────────────
    ls = exp.latency_stats
    def stat_row(label, key):
        s = ls.get(key, {})
        if not s.get("available"):
            return f"<tr><td>{label}</td><td colspan=6 style='color:#6b7280'>NOT AVAILABLE</td></tr>"
        return (f"<tr><td>{label}</td>"
                f"<td>{fmt_ms(s.get('min'))}</td>"
                f"<td>{fmt_ms(s.get('mean'))}</td>"
                f"<td>{fmt_ms(s.get('median'))}</td>"
                f"<td>{fmt_ms(s.get('p95'))}</td>"
                f"<td>{fmt_ms(s.get('p99'))}</td>"
                f"<td>{fmt_ms(s.get('max'))}</td></tr>")

    latency_rows = (
        stat_row("End-to-End", "e2e") +
        stat_row("ASR (stop→final)", "asr") +
        stat_row("Pipeline NMT+TTS", "pipeline_nmt_tts") +
        stat_row("Audio Delivery", "audio_delivery") +
        stat_row("PCM Send", "send")
    )

    # ── Turn-by-turn latency table (persistent session analysis) ──────
    by_turn = exp.latency_by_turn
    turn_rows = ""
    for tn_str in sorted(by_turn.keys(), key=lambda x: int(x)):
        td = by_turn[tn_str]
        success_rate = td['success_count'] / max(td['count'], 1) * 100
        turn_rows += (f"<tr><td>Turn {td['turn']}</td>"
                      f"<td>{td['count']}</td>"
                      f"<td style='color:{pct_color(success_rate)}'>{success_rate:.0f}%</td>"
                      f"<td>{fmt_ms(td.get('median'))}</td>"
                      f"<td>{fmt_ms(td.get('p95'))}</td></tr>")

    # ── Per-turn event log ────────────────────────────────────────────
    turn_log_rows = ""
    for t in turns:
        stage = t.failure_stage.value if t.failure_stage else ""
        errs  = "; ".join(t.errors[:2]) if t.errors else ""
        turn_log_rows += (
            f"<tr>"
            f"<td>{t.turn_number}</td>"
            f"<td>{t.source_language}</td>"
            f"<td>{t.asr_detected_language or '—'}</td>"
            f"<td>{status_badge(t.status.value)}</td>"
            f"<td>{fmt_ms(t.asr_latency_ms)}</td>"
            f"<td>{fmt_ms(t.pipeline_latency_ms)}</td>"
            f"<td>{fmt_ms(t.total_e2e_latency_ms)}</td>"
            f"<td>{t.audio_bytes_received:,}</td>"
            f"<td style='color:#ef4444;font-size:10px'>{stage or errs}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>V2 WebSocket Reliability Report — {exp.run_label}</title>
<style>
  :root {{
    --bg: #07101f; --surface: #0c1524; --border: #1a2840;
    --accent: #3b82f6; --green: #22c55e; --amber: #f59e0b; --red: #ef4444;
    --text: #e2e8f0; --muted: #64748b; --subtle: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
          padding: 24px; font-size: 14px; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; color: var(--subtle); margin: 28px 0 12px; border-bottom: 1px solid var(--border);
        padding-bottom: 6px; }}
  .meta {{ color: var(--muted); font-size: 12px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
           padding: 16px 24px; min-width: 160px; }}
  .card-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                 letter-spacing: .5px; margin-bottom: 6px; }}
  .card-val {{ font-size: 32px; font-weight: 800; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 24px; }}
  th {{ background: #0a1422; padding: 10px 12px; font-size: 11px; text-align: left;
        color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }}
  td {{ padding: 8px 12px; font-size: 12px; border-top: 1px solid var(--border); }}
  tr:hover td {{ background: #0f1a2e; }}
  .chart-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
  @media (max-width: 1024px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>🛡️ V2 WebSocket Reliability + Latency Report</h1>
<div class="meta">
  Run: <strong>{exp.run_label}</strong> &nbsp;|&nbsp;
  Started: {run_start} &nbsp;|&nbsp;
  Duration: ~{duration_min} min &nbsp;|&nbsp;
  Run ID: <code>{exp.run_id}</code>
</div>

<div class="note">
  ⓘ <strong>What "Clean Success" means here:</strong>
  ASR transcript received → audio_chunk(s) received → audio_end received → received PCM is decodable.
  Actual speaker playback cannot be verified from a Python test runner (requires a real browser AudioContext).
</div>

<div class="cards">
  <div class="card">
    <div class="card-label">Total Turns</div>
    <div class="card-val">{exp.total_turns}</div>
  </div>
  <div class="card">
    <div class="card-label">Clean Success</div>
    <div class="card-val" style="color:var(--green)">{exp.clean_successes}</div>
  </div>
  <div class="card">
    <div class="card-label">Partial</div>
    <div class="card-val" style="color:var(--amber)">{exp.partial_successes}</div>
  </div>
  <div class="card">
    <div class="card-label">Failures</div>
    <div class="card-val" style="color:var(--red)">{exp.failures}</div>
  </div>
  <div class="card">
    <div class="card-label">Clean Reliability</div>
    <div class="card-val" style="color:{pct_color(cr)}">{cr:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-label">User-Visible Success</div>
    <div class="card-val" style="color:{pct_color(uv)}">{uv:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Pipeline Success</div>
    <div class="card-val" style="color:{pct_color(ps)}">{ps:.1f}%</div>
  </div>
</div>

<h2>🌐 Per-Language Breakdown</h2>
<table>
  <thead>
    <tr>
      <th>Language</th><th>Turns</th><th>Reliability</th><th>Pipeline OK</th>
      <th>✅ Clean</th><th>⚠️ Partial</th><th>❌ Failed</th>
      <th>Median E2E</th><th>Median ASR</th><th>Median Pipe</th><th>Primary Failure</th>
    </tr>
  </thead>
  <tbody>{lang_rows}</tbody>
</table>

<h2>📊 Visualizations</h2>
<div class="chart-grid">
    <div class="chart-container">
        <h3 style="margin-bottom: 12px; font-size: 14px; color: var(--muted);">Global Median Latencies</h3>
        <canvas id="globalLatencyChart"></canvas>
    </div>
    <div class="chart-container">
        <h3 style="margin-bottom: 12px; font-size: 14px; color: var(--muted);">Clean Reliability by Language (%)</h3>
        <canvas id="langRelChart"></canvas>
    </div>
</div>
<div class="chart-container">
    <h3 style="margin-bottom: 12px; font-size: 14px; color: var(--muted);">Latency Breakdown by Language</h3>
    <canvas id="langCompareChart" style="max-height: 400px;"></canvas>
</div>

<h2>📊 Stage Latency Statistics</h2>
<table>
  <thead>
    <tr><th>Stage</th><th>Min</th><th>Mean</th><th>Median</th><th>P95</th><th>P99</th><th>Max</th></tr>
  </thead>
  <tbody>{latency_rows}</tbody>
</table>

<h2>📈 Persistent Session Analysis (Latency by Turn Number)</h2>
<p style="color:var(--muted);font-size:12px;margin-bottom:10px">
  Each WebSocket connection serves 10 turns. This table shows whether latency or reliability
  degrades as the connection ages.
</p>
<table>
  <thead>
    <tr><th>Turn #</th><th>Conversations</th><th>Success Rate</th><th>Median E2E</th><th>P95 E2E</th></tr>
  </thead>
  <tbody>{turn_rows}</tbody>
</table>

<h2>🔥 Failure Taxonomy</h2>
<table>
  <thead><tr><th>Stage</th><th>Count</th></tr></thead>
  <tbody>{failure_rows or "<tr><td colspan=2 style='color:var(--green)'>No failures</td></tr>"}</tbody>
</table>

<h2>📋 Turn-by-Turn Event Log</h2>
<table>
  <thead>
    <tr>
      <th>Turn</th><th>Src Lang</th><th>Detected</th><th>Status</th>
      <th>ASR ms</th><th>Pipeline ms</th><th>E2E ms</th><th>Audio Bytes</th><th>Error</th>
    </tr>
  </thead>
  <tbody>{turn_log_rows}</tbody>
</table>

<p style="color:var(--muted);font-size:11px;margin-top:20px">
  Generated by varta.ai V2 WebSocket Reliability Test Runner · {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</p>

<script>
{json.dumps([conv.language for conv in exp.conversations])}.then = null; // Just to make sure json loads isn't syntax error without var
const langLabels = {json.dumps([conv.language for conv in exp.conversations])};
const globalLabels = ["PCM Send", "ASR", "NMT+TTS", "Audio Delivery", "End-to-End"];
const globalValues = [
    {ls.get("send", {}).get("median", 0) or 0},
    {ls.get("asr", {}).get("median", 0) or 0},
    {ls.get("pipeline_nmt_tts", {}).get("median", 0) or 0},
    {ls.get("audio_delivery", {}).get("median", 0) or 0},
    {ls.get("e2e", {}).get("median", 0) or 0}
];

const langAsr = {json.dumps([conv.median_asr_latency_ms or 0 for conv in exp.conversations])};
const langPipe = {json.dumps([conv.median_pipeline_latency_ms or 0 for conv in exp.conversations])};
const langAudio = {json.dumps([(conv.median_e2e_latency_ms or 0) - (conv.median_asr_latency_ms or 0) - (conv.median_pipeline_latency_ms or 0) for conv in exp.conversations])};
const langRel = {json.dumps([conv.reliability_pct for conv in exp.conversations])};

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1a2840';

new Chart(document.getElementById('globalLatencyChart'), {{
    type: 'bar',
    data: {{
        labels: globalLabels,
        datasets: [{{
            label: 'Median Latency (ms)',
            data: globalValues,
            backgroundColor: ['#64748b', '#3b82f6', '#8b5cf6', '#14b8a6', '#f59e0b'],
            borderRadius: 4
        }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('langCompareChart'), {{
    type: 'line',
    data: {{
        labels: langLabels,
        datasets: [
            {{ label: 'ASR (ms)', data: langAsr, borderColor: '#3b82f6', backgroundColor: '#3b82f6', fill: false, tension: 0.3 }},
            {{ label: 'NMT+TTS (ms)', data: langPipe, borderColor: '#8b5cf6', backgroundColor: '#8b5cf6', fill: false, tension: 0.3 }},
            {{ label: 'Remaining E2E (ms)', data: langAudio, borderColor: '#f59e0b', backgroundColor: '#f59e0b', fill: false, tension: 0.3 }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{ title: {{ display: true, text: 'Total Latency (ms)' }} }}
        }}
    }}
}});

new Chart(document.getElementById('langRelChart'), {{
    type: 'line',
    data: {{
        labels: langLabels,
        datasets: [{{
            label: 'Reliability %',
            data: langRel,
            borderColor: '#22c55e',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            fill: true,
            tension: 0.3,
            pointBackgroundColor: '#22c55e'
        }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ min: 0, max: 100 }} }} }}
}});
</script>
</body>
</html>"""


# ── Convenience: print final summary to stdout ────────────────────────────────

def print_final_summary(exp: ExperimentResult) -> None:
    """Print the final report summary to stdout."""
    print()
    print("=" * 65)
    print("  V2 WebSocket Reliability + Latency — Experiment Results")
    print("=" * 65)
    print(f"  Run:               {exp.run_label}")
    print(f"  Total turns:       {exp.total_turns}")
    print()
    print(f"  ✅  Clean success:       {exp.clean_successes:3d}  ({exp.clean_reliability_pct:.1f}%)")
    print(f"  ⚠️   Partial success:     {exp.partial_successes:3d}")
    print(f"  ❌  Failures:            {exp.failures:3d}")
    print()
    print(f"  Clean reliability:        {exp.clean_reliability_pct:.1f}%")
    print(f"  User-visible success:     {exp.user_visible_success_pct:.1f}%")
    print(f"  Pipeline success (ASR OK):{exp.pipeline_success_pct:.1f}%")
    print()

    bd = exp.to_dict().get("failure_breakdown", {})
    print("  Failure Breakdown:")
    for stage, count in bd.items():
        if count > 0:
            print(f"    {stage.upper():20s} {count}")

    print()
    ls = exp.latency_stats
    print("  Latency Statistics (ms):")
    print(f"    {'Stage':<25} {'Median':>8} {'P95':>8} {'Max':>8}")
    for name, key in [
        ("End-to-End",     "e2e"),
        ("ASR",            "asr"),
        ("NMT+TTS Pipeline","pipeline_nmt_tts"),
        ("Audio Delivery",  "audio_delivery"),
    ]:
        s = ls.get(key, {})
        if s.get("available"):
            print(f"    {name:<25} {s['median']:>7}  {s['p95']:>7}  {s['max']:>7}")
        else:
            print(f"    {name:<25} {'N/A':>7}  {'N/A':>7}  {'N/A':>7}")

    print()
    print("  Per-Language Summary:")
    print(f"    {'Lang':<10} {'Turns':>6} {'OK%':>6} {'Median E2E':>12} {'Primary Fail'}")
    for conv in exp.conversations:
        print(f"    {conv.language:<10} {conv.total_turns:>6} "
              f"{conv.reliability_pct:>5.1f}% "
              f"{str(conv.median_e2e_latency_ms or '—') + 'ms':>12}  "
              f"{conv.primary_failure_stage or '—'}")
    print("=" * 65)
