import json
import re
from pathlib import Path

def regenerate(run_dir: Path):
    summary_file = run_dir / 'summary.json'
    html_file = run_dir / 'report.html'
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ls = data.get('latency_stats', {})
    convs = data.get('conversations', [])
    
    globalLabels = ["PCM Send", "ASR", "NMT+TTS", "Audio Delivery", "End-to-End"]
    globalValues = [
        ls.get("send", {}).get("median", 0) or 0,
        ls.get("asr", {}).get("median", 0) or 0,
        ls.get("pipeline_nmt_tts", {}).get("median", 0) or 0,
        ls.get("audio_delivery", {}).get("median", 0) or 0,
        ls.get("e2e", {}).get("median", 0) or 0
    ]
    
    langLabels = [c['language'] for c in convs]
    langAsr = [c.get('median_asr_latency_ms', 0) or 0 for c in convs]
    langPipe = [c.get('median_pipeline_latency_ms', 0) or 0 for c in convs]
    langAudio = [(c.get('median_e2e_latency_ms', 0) or 0) - (c.get('median_asr_latency_ms', 0) or 0) - (c.get('median_pipeline_latency_ms', 0) or 0) for c in convs]
    langRel = [c.get('reliability_pct', 0) for c in convs]
    
    chart_script = f"""
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const langLabels = {json.dumps(langLabels)};
const globalLabels = {json.dumps(globalLabels)};
const globalValues = {json.dumps(globalValues)};
const langAsr = {json.dumps(langAsr)};
const langPipe = {json.dumps(langPipe)};
const langAudio = {json.dumps(langAudio)};
const langRel = {json.dumps(langRel)};

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
"""

    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    
    html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL)
    html = html.replace('</body>', '')
    html = html.replace('</html>', chart_script + '\n</html>')
        
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Successfully injected charts into {html_file}")

if __name__ == '__main__':
    regenerate(Path('test/results/run_20260813_233115'))
