const TIMINGS = {
  rest: {
    bro: { upl: 1500, wat: 5000, prs: 50, dec: 180 },
    srv: { ses: 20, bld: 10, res: 15, sav: 15, log: 5 },
    asr: { tcp: 100, api: 1200, prs: 50 },
    nmt: { tcp: 100, api: 300, prs: 20 },
    tts: { tcp: 100, api: 3200, prs: 80 }
  },
  ws: {
    bro: { upl: 100, wat: 2200, prs: 10, dec: 20 },
    srv: { ses: 20, bld: 10, res: 15, sav: 15, log: 5 },
    asr: { tcp: 50, api: 780, prs: 20 },
    nmt: { tcp: 50, api: 350, prs: 20 },
    tts: { tcp: 50, api: 850, prs: 50 }
  },
  hybrid: {
    bro: { upl: 17161, wat: 17161, prs: 174, dec: 188 },
    srv: { ses: 64, bld: 0, res: 762, sav: 64, log: 0 },
    asr: { tcp: 0, api: 0, prs: 0 },
    nmt: { tcp: 1, api: 382, prs: 0 },
    tts: { tcp: 0, api: 0, prs: 0 }
  }
};

let currentArch = 'hybrid';
let active = false;

document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    if(active) return;
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    currentArch = t.dataset.arch;
    reset();
  });
});

document.getElementById('btn-reset').addEventListener('click', reset);
document.getElementById('btn-start').addEventListener('click', simulate);

function setMetric(id, suffix, val, maxVal) {
  document.getElementById(`v-${id}-${suffix}`).textContent = val + 'ms';
  const bar = document.getElementById(`v-${id}-${suffix}`).previousElementSibling;
  const pct = Math.min((val / (maxVal || 2000)) * 100, 100);
  bar.style.width = (val === 0 ? 0 : Math.max(pct, 5)) + '%';
}

function resetMetrics(suffix) {
  const ids = [
    'bro-upl', 'bro-wat', 'bro-prs', 'bro-dec',
    'srv-ses', 'srv-bld', 'srv-res', 'srv-sav', 'srv-log',
    'asr-tcp', 'asr-api', 'asr-prs',
    'nmt-tcp', 'nmt-api', 'nmt-prs',
    'tts-tcp', 'tts-api', 'tts-prs'
  ];
  ids.forEach(id => setMetric(id, suffix, 0));
  document.getElementById(`v-total-${suffix}`).textContent = '0ms';
}

function reset() {
  active = false;
  document.querySelectorAll('.node').forEach(n => n.classList.remove('active', 'done'));
  document.querySelectorAll('.line').forEach(l => l.classList.remove('active'));
  resetMetrics('ab');
  resetMetrics('ba');
  document.getElementById('btn-start').disabled = false;
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
function actNode(id) { document.getElementById(id).classList.add('active'); }
function actPath(id) { document.getElementById(id).classList.add('active'); }
function donNode(id) { 
  const el = document.getElementById(id);
  el.classList.remove('active'); el.classList.add('done'); 
}
function donPath(id) { document.getElementById(id).classList.remove('active'); }

async function simulate() {
  if (active) return;
  active = true;
  document.getElementById('btn-start').disabled = true;
  resetMetrics('ab');
  resetMetrics('ba');

  const t = TIMINGS[currentArch];

  // PHASE 1: Speaker A Speaks -> ASR -> Redis Buffer
  actNode('node-speaker-a');
  await delay(500);
  actPath('path-sa-asr');
  await delay(300);
  donNode('node-speaker-a');
  
  actNode('node-asr-top');
  setMetric('bro-upl', 'ab', t.bro.upl, 20000);
  setMetric('srv-ses', 'ab', t.srv.ses, 100);
  setMetric('asr-tcp', 'ab', t.asr.tcp, 200);
  setMetric('asr-api', 'ab', t.asr.api, 1500);
  setMetric('asr-prs', 'ab', t.asr.prs, 100);
  await delay(800);
  
  actPath('path-asr-srca');
  actPath('path-asr-trana');
  actNode('node-src-lang-a');
  actNode('node-tran-a1');
  await delay(400);
  donPath('path-sa-asr');
  donNode('node-asr-top');
  
  actPath('path-srca-redis');
  actPath('path-trana-redis');
  actNode('node-redis');
  setMetric('srv-sav', 'ab', t.srv.sav, 100);
  await delay(500);
  donNode('node-src-lang-a');
  donNode('node-tran-a1');
  donPath('path-asr-srca');
  donPath('path-asr-trana');
  donPath('path-srca-redis');
  donPath('path-trana-redis');
  
  // Pipeline pauses, wait for B
  await delay(1000);

  // PHASE 2: Speaker B Speaks -> ASR -> Redis Buffer
  actNode('node-speaker-b');
  await delay(500);
  actPath('path-sb-asr');
  await delay(300);
  donNode('node-speaker-b');

  actNode('node-asr-bot');
  setMetric('bro-upl', 'ba', t.bro.upl, 20000);
  setMetric('srv-ses', 'ba', t.srv.ses, 100);
  setMetric('asr-tcp', 'ba', t.asr.tcp, 200);
  setMetric('asr-api', 'ba', t.asr.api, 1500);
  setMetric('asr-prs', 'ba', t.asr.prs, 100);
  await delay(800);

  actPath('path-asr-srcb');
  actPath('path-asr-tranb');
  actNode('node-src-lang-b');
  actNode('node-tran-b1');
  await delay(400);
  donNode('node-asr-bot');
  donPath('path-sb-asr');

  actPath('path-srcb-redis');
  actPath('path-tranb-redis');
  setMetric('srv-sav', 'ba', t.srv.sav, 100);
  await delay(500);
  donNode('node-src-lang-b');
  donNode('node-tran-b1');
  donPath('path-asr-srcb');
  donPath('path-asr-tranb');
  donPath('path-srcb-redis');
  donPath('path-tranb-redis');

  // PHASE 3: CONCURRENT NMT
  // Redis fires outputs for both pipelines simultaneously
  actPath('path-redis-srcb-top');
  actPath('path-redis-trana-top');
  actNode('node-src-lang-b-top');
  actNode('node-tran-a2');

  actPath('path-redis-srca-bot');
  actPath('path-redis-tranb-bot');
  actNode('node-src-lang-a-bot');
  actNode('node-tran-b2');

  await delay(600);
  
  actPath('path-srcb-nmt-top');
  actPath('path-trana-nmt-top');
  actPath('path-srca-nmt-bot');
  actPath('path-tranb-nmt-bot');
  
  actNode('node-nmt-top');
  actNode('node-nmt-bot');
  
  setMetric('nmt-tcp', 'ab', t.nmt.tcp, 200);
  setMetric('nmt-api', 'ab', t.nmt.api, 500);
  setMetric('nmt-prs', 'ab', t.nmt.prs, 100);

  setMetric('nmt-tcp', 'ba', t.nmt.tcp, 200);
  setMetric('nmt-api', 'ba', t.nmt.api, 500);
  setMetric('nmt-prs', 'ba', t.nmt.prs, 100);
  
  await delay(800);

  donNode('node-src-lang-b-top');
  donNode('node-tran-a2');
  donPath('path-redis-srcb-top');
  donPath('path-redis-trana-top');
  donPath('path-srcb-nmt-top');
  donPath('path-trana-nmt-top');

  donNode('node-src-lang-a-bot');
  donNode('node-tran-b2');
  donPath('path-redis-srca-bot');
  donPath('path-redis-tranb-bot');
  donPath('path-srca-nmt-bot');
  donPath('path-tranb-nmt-bot');
  
  // PHASE 4: CONCURRENT TTS
  actPath('path-nmt-tts-top');
  actPath('path-nmt-tts-bot');
  actNode('node-tts-top');
  actNode('node-tts-bot');

  setMetric('tts-tcp', 'ab', t.tts.tcp, 200);
  setMetric('tts-api', 'ab', t.tts.api, 4000);
  setMetric('tts-prs', 'ab', t.tts.prs, 100);

  setMetric('tts-tcp', 'ba', t.tts.tcp, 200);
  setMetric('tts-api', 'ba', t.tts.api, 4000);
  setMetric('tts-prs', 'ba', t.tts.prs, 100);

  await delay(1000);
  donNode('node-nmt-top');
  donPath('path-nmt-tts-top');
  donNode('node-nmt-bot');
  donPath('path-nmt-tts-bot');

  // PHASE 5: Delivery to Listeners
  actPath('path-tts-lb');
  actPath('path-tts-la');
  actNode('node-listener-b');
  actNode('node-listener-a');

  setMetric('bro-wat', 'ab', t.bro.wat, 20000);
  setMetric('srv-bld', 'ab', t.srv.bld, 100);
  setMetric('srv-res', 'ab', t.srv.res, 1000);
  setMetric('srv-log', 'ab', t.srv.log, 50);
  setMetric('bro-prs', 'ab', t.bro.prs, 200);
  setMetric('bro-dec', 'ab', t.bro.dec, 300);

  setMetric('bro-wat', 'ba', t.bro.wat, 20000);
  setMetric('srv-bld', 'ba', t.srv.bld, 100);
  setMetric('srv-res', 'ba', t.srv.res, 1000);
  setMetric('srv-log', 'ba', t.srv.log, 50);
  setMetric('bro-prs', 'ba', t.bro.prs, 200);
  setMetric('bro-dec', 'ba', t.bro.dec, 300);

  const totalLat = currentArch === 'hybrid' ? 17564 : (t.asr.api + t.nmt.api + t.tts.api + t.bro.wat);
  document.getElementById('v-total-ab').textContent = totalLat + 'ms';
  document.getElementById('v-total-ba').textContent = totalLat + 'ms';

  await delay(1000);
  donNode('node-tts-top');
  donPath('path-tts-lb');
  donNode('node-listener-b');
  
  donNode('node-tts-bot');
  donPath('path-tts-la');
  donNode('node-listener-a');

  donNode('node-redis');
  active = false;
}
