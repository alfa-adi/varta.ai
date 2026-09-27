#!/usr/bin/env python3
"""
visualize_bottlenecks.py
────────────────────────
Reads pipeline_events from MongoDB for a completed test session,
produces 5 PNG charts in test/results/, and prints a bottleneck
diagnosis to stdout.

Usage:
    python visualize_bottlenecks.py <session_id>

Read-only against MongoDB. Does not modify any other file.
"""

import io
import os
import sys

# Force UTF-8 output on Windows (cp1252 can't handle arrows/emojis)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

COLORS = {
    "tcp_connect_ms": "#EF4444",   # red
    "ttfb_ms":        "#9CA3AF",   # grey (TTFB = full round trip, not just upload)
    "asr_ms":         "#3B82F6",   # blue
    "nmt_ms":         "#8B5CF6",   # purple
    "tts_ms":         "#10B981",   # green
    "download_ms":    "#F59E0B",   # amber
}

STAGE_LABELS = {
    "tcp_connect_ms": "TCP Connect",
    "ttfb_ms":        "TTFB (full round trip)",
    "asr_ms":         "ASR",
    "nmt_ms":         "NMT",
    "tts_ms":         "TTS",
    "download_ms":    "Download",
}

LANG_FULL = {
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "te": "Telugu",
    "ta": "Tamil", "ur": "Urdu", "gu": "Gujarati", "kn": "Kannada",
    "or": "Odia", "ml": "Malayalam", "pa": "Punjabi", "as": "Assamese",
    "sd": "Sindhi",
}

STAGES_ALL = ["tcp_connect_ms", "ttfb_ms", "asr_ms", "nmt_ms", "tts_ms", "download_ms"]
STAGES_NO_TCP = ["ttfb_ms", "asr_ms", "nmt_ms", "tts_ms", "download_ms"]

BASE_RESULTS_DIR = "test/results"

# ── Config ────────────────────────────────────────────────────────────────────

MONGO_URL     = os.getenv("MONGO_URL")
MONGO_TEST_DB = os.getenv("MONGO_TEST_DB", "varta_test_data")


def main():
    # ── Argument validation ───────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python visualize_bottlenecks.py <session_id>")
        print("  Produces 5 PNG charts in test/results/<date>_<session>/ and prints diagnosis.")
        sys.exit(1)

    session_id = sys.argv[1]

    if not MONGO_URL:
        print("ERROR: MONGO_URL environment variable is not set.")
        sys.exit(1)

    # ── Connect to MongoDB ────────────────────────────────────────
    from pymongo import MongoClient

    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB — {e}")
        sys.exit(1)

    db = client[MONGO_TEST_DB]

    # ── Validate session ──────────────────────────────────────────
    session = db["test_sessions"].find_one({"session_id": session_id})
    if not session:
        print(f"ERROR: Session ID '{session_id}' not found in test_sessions.")
        sys.exit(1)

    # ── Single query: fetch all pipeline_events ───────────────────
    docs = list(db.pipeline_events.find(
        {"session_id": session_id, "success": True},
        {
            "source_language": 1,
            "browser.tcp_connect_ms": 1,
            "browser.ttfb_ms": 1,
            "browser.upload_ms": 1,  # backwards compat with old data
            "browser.download_ms": 1,
            "browser.total_ms": 1,
            "server.asr_ms": 1,
            "server.nmt_ms": 1,
            "server.tts_ms": 1,
            "server.sarvam_tcp_ms": 1,
            "_id": 0,
        }
    ))

    if not docs:
        print(f"ERROR: Session '{session_id}' has 0 successful events.")
        sys.exit(1)

    # ── Flatten into DataFrame ────────────────────────────────────
    flat = []
    for doc in docs:
        browser = doc.get("browser", {})
        server = doc.get("server", {})
        flat.append({
            "source_language":  doc.get("source_language"),
            "tcp_connect_ms":   browser.get("tcp_connect_ms"),   # may be None
            "ttfb_ms":          browser.get("ttfb_ms") or browser.get("upload_ms"),  # compat
            "download_ms":      browser.get("download_ms"),
            "total_ms":         browser.get("total_ms"),
            "asr_ms":           server.get("asr_ms"),
            "nmt_ms":           server.get("nmt_ms"),
            "tts_ms":           server.get("tts_ms"),
            "sarvam_tcp_ms":    server.get("sarvam_tcp_ms"),
        })

    df = pd.DataFrame(flat)
    total_turns = len(df)

    # ── Build dated output folder ─────────────────────────────────
    # e.g. test/results/2026-07-10_1163679f/
    started_at = session.get("started_at")
    if started_at:
        date_str = started_at.strftime("%Y-%m-%d")
    else:
        from datetime import date as _date
        date_str = _date.today().isoformat()
    short_id = session_id[:8]
    OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, f"{date_str}_{short_id}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoaded {total_turns} successful events for session {session_id}")
    print(f"Generating charts in {OUTPUT_DIR}/...\n")

    # ══════════════════════════════════════════════════════════════
    # Chart 1 — Stage Breakdown Bar
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    stage_means = {}
    for s in STAGES_ALL:
        stage_means[s] = df[s].mean()  # .mean() skips NaN

    x_labels = [STAGE_LABELS[s] for s in STAGES_ALL]
    x_vals   = [stage_means[s] if pd.notnull(stage_means[s]) else 0 for s in STAGES_ALL]
    x_colors = [COLORS[s] for s in STAGES_ALL]

    bars = ax.bar(x_labels, x_vals, color=x_colors, edgecolor="white", linewidth=0.5)

    # Label each bar
    for i, (bar, stage) in enumerate(zip(bars, STAGES_ALL)):
        val = stage_means[stage]
        if pd.notnull(val) and val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{val:.0f}ms", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # TCP annotation
    tcp_non_null = df["tcp_connect_ms"].notnull().sum()
    tcp_bar = bars[0]
    ax.text(tcp_bar.get_x() + tcp_bar.get_width() / 2,
            max(tcp_bar.get_height() / 2, 10),
            f"TCP: avg of {tcp_non_null} non-null\nvalues out of {total_turns} total turns",
            ha="center", va="center", fontsize=7, color="white",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.3"))

    # Total mean line
    total_mean = df["total_ms"].mean()
    ax.axhline(total_mean, color="#6B7280", linestyle="--", linewidth=1.5)
    ax.text(len(STAGES_ALL) - 0.5, total_mean + 40, f"Total Mean: {total_mean:.0f}ms",
            ha="right", va="bottom", color="#6B7280", fontweight="bold", fontsize=9)

    ax.set_title("Chart 1 — Stage Breakdown (Mean Latency per Stage)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart1_stage_breakdown.png", dpi=150)
    plt.close(fig)
    print("  Saved chart1_stage_breakdown.png")

    # ══════════════════════════════════════════════════════════════
    # Chart 2 — Per-Stage Box Plots
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)

    for ax, stage in zip(axes, STAGES_NO_TCP):
        valid = df[stage].dropna()
        if len(valid) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#9CA3AF")
            ax.set_xticks([])
        else:
            bp = ax.boxplot(valid, patch_artist=True, widths=0.6)
            bp["boxes"][0].set_facecolor(COLORS[stage])
            bp["boxes"][0].set_alpha(0.8)
            bp["medians"][0].set_color("white")
            bp["medians"][0].set_linewidth(2)
            ax.set_xticks([])

        ax.set_title(STAGE_LABELS[stage], fontsize=11, fontweight="bold")
        ax.set_xlabel(f"n={len(valid)}", fontsize=9, color="#6B7280")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Chart 2 — Per-Stage Box Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart2_stage_boxplots.png", dpi=150)
    plt.close(fig)
    print("  Saved chart2_stage_boxplots.png")

    # ══════════════════════════════════════════════════════════════
    # Chart 3 — Total Latency per Language
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))

    lang_medians = df.groupby("source_language")["total_ms"].median().sort_values(ascending=False)
    sorted_langs = lang_medians.index.tolist()

    box_data = [df[df["source_language"] == lang]["total_ms"].dropna().values for lang in sorted_langs]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
    for box in bp["boxes"]:
        box.set_facecolor("#3B82F6")
        box.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    ax.set_xticklabels([LANG_FULL.get(l, l) for l in sorted_langs], rotation=30, ha="right")
    ax.set_title("Chart 3 — Total Latency per Language (sorted by median)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart3_latency_by_language.png", dpi=150)
    plt.close(fig)
    print("  Saved chart3_latency_by_language.png")

    # ══════════════════════════════════════════════════════════════
    # Chart 4 — Stage Breakdown per Language (Horizontal Stacked)
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 8))

    lang_stage_means = df.groupby("source_language")[STAGES_NO_TCP].mean()
    lang_stage_means["total_sum"] = lang_stage_means.sum(axis=1)
    lang_stage_means = lang_stage_means.sort_values("total_sum", ascending=True)
    lang_stage_means = lang_stage_means.drop(columns="total_sum")

    y_labels = [LANG_FULL.get(l, l) for l in lang_stage_means.index]
    y_pos = range(len(y_labels))

    lefts = pd.Series([0.0] * len(lang_stage_means), index=lang_stage_means.index)
    for stage in STAGES_NO_TCP:
        vals = lang_stage_means[stage].fillna(0)
        ax.barh(y_pos, vals, left=lefts, color=COLORS[stage], label=STAGE_LABELS[stage],
                edgecolor="white", linewidth=0.5)
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Chart 4 — Stage Breakdown per Language", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart4_stage_per_language.png", dpi=150)
    plt.close(fig)
    print("  Saved chart4_stage_per_language.png")

    # ══════════════════════════════════════════════════════════════
    # Chart 5 — TCP Connection Analysis
    # ══════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean TCP connect time per language (non-null only)
    tcp_by_lang = df.groupby("source_language")["tcp_connect_ms"]
    tcp_means   = tcp_by_lang.mean().dropna()
    tcp_counts  = tcp_by_lang.apply(lambda x: x.notnull().sum())

    if tcp_means.empty:
        ax1.text(0.5, 0.5,
                 "No new TCP connections\nobserved in this session\n(all turns reused warm connection)",
                 ha="center", va="center", fontsize=12, color="#6B7280",
                 transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
        for spine in ax1.spines.values():
            spine.set_visible(False)
    else:
        bars = ax1.bar(range(len(tcp_means)), tcp_means.values, color=COLORS["tcp_connect_ms"],
                       edgecolor="white", linewidth=0.5)
        ax1.set_xticks(range(len(tcp_means)))
        ax1.set_xticklabels([LANG_FULL.get(l, l) for l in tcp_means.index], rotation=30, ha="right")
        ax1.set_ylabel("TCP Connect Time (ms)")
        for i, bar in enumerate(bars):
            lang = tcp_means.index[i]
            n = tcp_counts.get(lang, 0)
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"n={n}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_title("Mean TCP Connect Time", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Connection reuse percentage per language
    total_per_lang = df.groupby("source_language").size()
    null_per_lang  = df.groupby("source_language")["tcp_connect_ms"].apply(lambda x: x.isnull().sum())
    reuse_pct = (null_per_lang / total_per_lang * 100).fillna(100)
    reuse_pct = reuse_pct.sort_values(ascending=True)

    bars2 = ax2.bar(range(len(reuse_pct)), reuse_pct.values, color="#10B981",
                    edgecolor="white", linewidth=0.5)
    ax2.axhline(100, color="#6B7280", linestyle="--", linewidth=1)
    ax2.set_xticks(range(len(reuse_pct)))
    ax2.set_xticklabels([LANG_FULL.get(l, l) for l in reuse_pct.index], rotation=30, ha="right")
    ax2.set_ylabel("Connection Reuse (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Connection Reuse Percentage", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Chart 5 — TCP Connection Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart5_tcp_analysis.png", dpi=150)
    plt.close(fig)
    print("  Saved chart5_tcp_analysis.png")

    # ══════════════════════════════════════════════════════════════
    # Bottleneck Diagnosis — stdout
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("BOTTLENECK DIAGNOSIS")
    print("=" * 60)
    print()
    print(f"Session avg total latency: {total_mean:.0f}ms")
    print()
    print("Stages ranked by avg cost:")

    # Build ranked list
    ranked = []
    for stage in STAGES_ALL:
        val = stage_means[stage]
        if pd.notnull(val):
            ranked.append((stage, val))
    ranked.sort(key=lambda x: x[1], reverse=True)

    for stage, val in ranked:
        pct = (val / total_mean * 100) if total_mean > 0 else 0
        label = STAGE_LABELS[stage]
        suffix = ""
        if stage == "tcp_connect_ms":
            suffix = f", N={tcp_non_null} turns only"
        print(f"  {label:<16} {val:>6.0f}ms  ({pct:>2.0f}% of total{suffix})")

    print()
    print("Diagnosis:")

    issues_found = False

    # ASR check
    asr_val = stage_means.get("asr_ms", float("nan"))
    if pd.notnull(asr_val) and asr_val > 1200:
        issues_found = True
        print(f"  Primary bottleneck: ASR inference ({asr_val:.0f}ms avg).")

    # Sarvam TCP check
    sarvam_tcp_val = df["sarvam_tcp_ms"].mean()
    if pd.notnull(sarvam_tcp_val) and sarvam_tcp_val > 80:
        issues_found = True
        print(f"  server.sarvam_tcp_ms = {sarvam_tcp_val:.0f}ms — Render->Sarvam TCP reconnection per call.")
        print("  Fix (Step 7): httpx AsyncClient with connection reuse across requests.")
        print("  Expected saving: 300-600ms per turn.")
        print()

    # TTFB check (was previously mislabelled as "Upload")
    ttfb_val = stage_means.get("ttfb_ms", float("nan"))
    if pd.notnull(ttfb_val) and ttfb_val > 4000:
        issues_found = True
        print(f"  TTFB is elevated ({ttfb_val:.0f}ms). Includes ASR + NMT + TTS + network.")
        print("  Check individual stage latencies to identify the bottleneck.")
        print()

    # Download check
    download_val = stage_means.get("download_ms", float("nan"))
    if pd.notnull(download_val) and download_val > 200:
        issues_found = True
        print(f"  Download is elevated ({download_val:.0f}ms).")
        print("  Fix: implement raw binary TTS response streaming.")
        print()
    elif pd.notnull(download_val):
        print(f"  Download is normal ({download_val:.0f}ms). No action needed.")
        print()

    if not issues_found:
        print("  No single dominant bottleneck. All stages within expected range.")

    print()
    print(f"Charts saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
