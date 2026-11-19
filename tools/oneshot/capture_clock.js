// Smooth frames of an animated page on a virtual clock: node capture_clock.js URL OUTDIR SECONDS FPS WIDTH HEIGHT [WARMUP_S]
const { chromium } = require("/Users/suryakoritala/sigbound-forge/node_modules/playwright-core");
const fs = require("fs");
const [url, out, secs, fps, w, h, warm] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ channel: "chrome", headless: true });
  const p = await b.newPage({ viewport: { width: +w, height: +h } });
  await p.clock.install();
  await p.goto(url);
  for (let i = 0; i < (+warm || 0) * 30; i++) await p.clock.runFor(33);
  fs.mkdirSync(out, { recursive: true });
  const n = Math.round(+secs * +fps), step = Math.round(1000 / +fps);
  for (let i = 0; i < n; i++) { await p.clock.runFor(step); await p.screenshot({ path: `${out}/f${String(i).padStart(5, "0")}.png` }); }
  console.log(JSON.stringify({ frames: n }));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
