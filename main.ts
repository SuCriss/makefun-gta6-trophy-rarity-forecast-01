const fs = require('node:fs');
const path = require('node:path');

const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

function finite(x) {
  return typeof x === 'number' && isFinite(x);
}

function daysFromCivil(y, m, d) {
  const yy = y - (m <= 2 ? 1 : 0);
  const era = Math.floor((yy >= 0 ? yy : yy - 399) / 400);
  const yoe = yy - era * 400;
  const doy = Math.floor((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

function parseTs(s) {
  if (typeof s !== 'string') return NaN;
  const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?/.exec(s);
  if (!m) return NaN;
  return daysFromCivil(+m[1], +m[2], +m[3]) * 24 + +m[4] + +m[5] / 60 + +m[6] / 3600 + (m[7] ? +('0.' + m[7]) : 0) / 3600;
}

function logitP(p) {
  p = clamp(p, 1e-4, 1 - 1e-4);
  return Math.log(p / (1 - p));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-clamp(x, -40, 40)));
}

function median(arr) {
  const a = arr.filter(finite).slice().sort((x, y) => x - y);
  if (!a.length) return NaN;
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function buildObs(cs, nT) {
  const hist = Array.isArray(cs.history) ? cs.history : [];
  const obs = [];
  let prev = null;
  for (const h of hist) {
    if (!h || !Array.isArray(h.rates)) continue;
    let t = parseTs(h.slot);
    if (!finite(t)) t = parseTs(h.at);
    const N = +h.players_tracked;
    if (!finite(t) || !finite(N) || N <= 0) continue;
    const rates = new Array(nT);
    for (let i = 0; i < nT; i++) {
      const v = +h.rates[i];
      rates[i] = finite(v) ? clamp(v, 0, 100) : null;
    }
    let changed = !prev || prev.N !== N;
    if (!changed) {
      for (let i = 0; i < nT; i++) {
        const a = prev.rates[i];
        const b = rates[i];
        if (a === null || b === null) {
          if (a !== b) { changed = true; break; }
        } else if (Math.abs(a - b) > 1e-9) {
          changed = true;
          break;
        }
      }
    }
    if (changed) obs.push({ t, N, rates });
    prev = { t, N, rates };
  }
  for (const o of obs) {
    o.E = o.rates.map((r) => r === null ? null : Math.round(clamp(r * o.N / 100, 0, o.N)));
  }
  return obs;
}

function forecastN(obs, tStar, deltaDays) {
  const last = obs[obs.length - 1];
  const Nlast = last.N;
  const recent = obs.slice(-Math.min(12, obs.length));
  let slope = 0;
  let ws = 0;
  for (let j = 1; j < recent.length; j++) {
    const dt = recent[j].t - recent[j - 1].t;
    if (!(dt > 0)) continue;
    slope += dt * Math.log(recent[j].N / recent[j - 1].N) / dt;
    ws += dt;
  }
  if (ws > 0) slope /= ws;
  const priorFactor = clamp(1 + 1.1 * Math.sqrt(deltaDays) * (Nlast < 3000 ? 1.5 : 1), 1, 30);
  let obsFactor = 1;
  const dh = Math.max(0, tStar - last.t);
  if (ws > 0) {
    const eff = slope / (1 + dh / 60);
    obsFactor = clamp(Math.exp(eff * dh), 1, 50);
  }
  const factor = Math.exp(0.5 * Math.log(priorFactor) + 0.5 * Math.log(obsFactor));
  return clamp(Nlast * factor, Nlast, Nlast * 60);
}

function fitLogTrend(obs, i, tStar, t0, useE) {
  let pts = [];
  for (const o of obs) {
    if (useE) {
      if (o.E[i] !== null && o.E[i] > 0) pts.push(o);
    } else if (o.rates[i] !== null) {
      pts.push(o);
    }
  }
  if (!pts.length) return null;
  const lastT = pts[pts.length - 1].t;
  const recent = pts.filter((o) => lastT - o.t <= 48);
  if (recent.length >= 3) pts = recent;
  else if (pts.length > 8) pts = pts.slice(-8);
  if (pts.length < 2) return null;
  const xOf = (t) => Math.log(1 + Math.max(0, t - t0) / 24);
  const xs = pts.map((o) => xOf(o.t));
  const ys = pts.map((o) => useE ? Math.log(o.E[i]) : logitP(o.rates[i] / 100));
  const slopes = [];
  for (let a = 0; a < pts.length; a++) {
    for (let b = a + 1; b < pts.length; b++) {
      const dx = xs[b] - xs[a];
      if (dx > 1e-9) slopes.push((ys[b] - ys[a]) / dx);
    }
  }
  let slope = median(slopes);
  if (!finite(slope)) slope = 0;
  slope = clamp(slope, useE ? -3 : -2.5, useE ? 3 : 2.5);
  const damp = 1 / (1 + Math.max(0, tStar - lastT) / 120);
  let shift = slope * damp * (xOf(tStar) - xs[xs.length - 1]);
  shift = clamp(shift, useE ? -2.5 : -3, useE ? 2.5 : 3);
  return ys[ys.length - 1] + shift;
}

function estimateG(obs, i, lastT) {
  const recent = obs.slice(-Math.min(16, obs.length));
  let num = 0;
  let den = 0;
  for (let j = 1; j < recent.length; j++) {
    const dN = recent[j].N - recent[j - 1].N;
    const e0 = recent[j - 1].E[i];
    const e1 = recent[j].E[i];
    if (dN > 0 && e0 !== null && e1 !== null) {
      const g = clamp((e1 - e0) / dN, 0, 1);
      const age = lastT - recent[j].t;
      const w = dN * Math.exp(-age / 48);
      num += w * g;
      den += w;
    }
  }
  return { g: den > 0 ? num / den : NaN, w: den };
}

function jumpSigma(obs, i) {
  const vals = [];
  for (let j = 1; j < obs.length; j++) {
    const a = obs[j - 1].rates[i];
    const b = obs[j].rates[i];
    if (a !== null && b !== null) vals.push(logitP(b / 100) - logitP(a / 100));
  }
  if (!vals.length) return NaN;
  const m = vals.reduce((s, x) => s + x, 0) / vals.length;
  return Math.sqrt(vals.reduce((s, x) => s + (x - m) * (x - m), 0) / vals.length);
}

function fallbackRows(cs) {
  const trophies = Array.isArray(cs.trophies) ? cs.trophies : [];
  const hist = Array.isArray(cs.history) ? cs.history : [];
  const last = hist.length ? hist[hist.length - 1] : null;
  return trophies.map((tr, i) => {
    const v = last && Array.isArray(last.rates) && finite(+last.rates[i]) ? clamp(+last.rates[i], 0, 100) : 1;
    const r = Math.round(v * 10000) / 10000;
    return { id: finite(+tr.id) ? +tr.id : i + 1, q: [r, r, r, r, r] };
  });
}

function forecastCase(cs) {
  const trophies = Array.isArray(cs.trophies) ? cs.trophies : [];
  const nT = trophies.length;
  if (!nT) return { trophies: [] };
  const obs = buildObs(cs, nT);
  const tStar = parseTs(cs.target);
  const lastT = obs.length ? obs[obs.length - 1].t : (finite(tStar) ? tStar - 72 : 0);
  let deltaHours = finite(tStar) ? tStar - lastT : NaN;
  if (!(deltaHours > 0)) deltaHours = finite(+cs.horizon_minutes) ? +cs.horizon_minutes / 60 : 108;
  const deltaDays = deltaHours / 24;
  const t0 = obs.length ? obs[0].t : lastT - 24;
  const Nlast = obs.length ? obs[obs.length - 1].N : 10000;
  const Npred = obs.length ? forecastN(obs, finite(tStar) ? tStar : lastT + deltaHours, deltaDays) : Nlast * 2;

  const gInd = new Array(nT);
  const gW = new Array(nT);
  const jsig = new Array(nT);
  for (let i = 0; i < nT; i++) {
    const r = estimateG(obs, i, lastT);
    gInd[i] = r.g;
    gW[i] = r.w;
    jsig[i] = jumpSigma(obs, i);
  }
  const tierMed = {};
  for (const tier of ['platinum', 'gold', 'silver', 'bronze']) {
    const arr = [];
    for (let i = 0; i < nT; i++) if (trophies[i].tier === tier && finite(gInd[i])) arr.push(gInd[i]);
    tierMed[tier] = median(arr);
  }
  const pooled = median(gInd);
  const jsigMed = median(jsig.filter((v) => finite(v) && v > 0));
  const gaps = [];
  for (let j = 1; j < obs.length; j++) gaps.push(obs[j].t - obs[j - 1].t);
  const avgGap = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 1;
  const nFuture = Math.max(0.5, deltaHours / Math.max(0.5, avgGap));

  const rows = [];
  for (let i = 0; i < nT; i++) {
    let lastRate = null;
    let lastE = null;
    for (let j = obs.length - 1; j >= 0; j--) {
      if (obs[j].rates[i] !== null) {
        lastRate = obs[j].rates[i];
        lastE = obs[j].E[i];
        break;
      }
    }
    if (lastRate === null) {
      lastRate = 1;
      lastE = Nlast / 100;
    }
    if (lastE === null) lastE = lastRate * Nlast / 100;

    const tier = trophies[i].tier;
    let prior = finite(tierMed[tier]) ? tierMed[tier] : pooled;
    if (!finite(prior)) prior = lastRate / 100;
    let g = finite(gInd[i]) ? gInd[i] : prior;
    if (finite(gInd[i])) {
      const ew = clamp(gW[i] / 10, 0, 3);
      g = (ew * gInd[i] + 2 * prior) / (ew + 2);
    }
    g = clamp(g, 0, 1);

    const rate1 = clamp(g + (lastE - g * Nlast) / Npred, 0, 1);
    const mu2 = fitLogTrend(obs, i, finite(tStar) ? tStar : lastT + deltaHours, t0, false);
    const logE = fitLogTrend(obs, i, finite(tStar) ? tStar : lastT + deltaHours, t0, true);
    const rate3 = finite(logE) ? clamp(Math.exp(logE) / Npred * 100, 0, 100) : lastRate;
    const mu4 = logitP(lastRate / 100);
    const mu1 = logitP(rate1);
    const mu3 = logitP(rate3 / 100);
    const mu2v = finite(mu2) ? mu2 : mu4;
    let w1 = 0.55, w2 = 0.25, w3 = 0.1, w4 = 0.1;
    if (obs.length < 4) {
      w1 = 0.25;
      w2 = 0.15;
      w3 = 0.05;
      w4 = 0.55;
    }
    const mu = (w1 * mu1 + w2 * mu2v + w3 * mu3 + w4 * mu4) / (w1 + w2 + w3 + w4);

    const sigBase = 0.12 + 0.27 * Math.sqrt(deltaDays);
    const jump = finite(jsig[i]) ? jsig[i] : (finite(jsigMed) ? jsigMed : 0.15);
    const jterm = Math.min(0.85, jump * Math.sqrt(nFuture) * 0.35);
    const cterm = 0.5 / Math.sqrt(lastE + 1);
    let sigma = Math.sqrt(sigBase * sigBase + jterm * jterm + cterm * cterm + 0.0016);
    sigma = clamp(sigma, 0.18, 2);

    const q = Z.map((z) => Math.round(clamp(100 * sigmoid(mu + z * sigma), 0, 100) * 10000) / 10000);
    for (let k = 1; k < 5; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    rows.push({ id: finite(+trophies[i].id) ? +trophies[i].id : i + 1, q });
  }
  return { trophies: rows };
}

function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR || '/input';
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || '/output';
  fs.mkdirSync(outDir, { recursive: true });
  const names = fs.readdirSync(inDir).filter((n) => n.endsWith('.in')).sort();
  for (const name of names) {
    let out;
    try {
      const cs = JSON.parse(fs.readFileSync(path.join(inDir, name), 'utf8'));
      out = forecastCase(cs);
    } catch (e) {
      out = { trophies: [] };
    }
    if (!out.trophies || !out.trophies.length) {
      try {
        const cs = JSON.parse(fs.readFileSync(path.join(inDir, name), 'utf8'));
        out = { trophies: fallbackRows(cs) };
      } catch (e) {
        out = { trophies: [] };
      }
    }
    fs.writeFileSync(path.join(outDir, path.basename(name, '.in') + '.out'), JSON.stringify(out) + '\n');
  }
}

main();
