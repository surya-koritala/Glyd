// A game file as a player would meet it: node gametest.js FILE OUTDIR. The title screen, then #autoplay for 40 s of real time, then a player's
// keys (Enter, arrows, Space) for 10 s; script errors and screenshots along the way; the page's own numbers it shows (score) read from the screenshots by eye.
const { chromium } = require("/Users/suryakoritala/sigbound-forge/node_modules/playwright-core");
const fs = require("fs"), [file, out] = process.argv.slice(2);
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ channel: "chrome", headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
  const errors = [];
  const page = async (hash) => {
    const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
    p.on("pageerror", e => errors.push(`${hash || "title"}: ${e.message}`));
    p.on("console", m => m.type() === "error" && errors.push(`${hash || "title"} console: ${m.text()}`));
    await p.goto("file://" + file + hash);
    return p;
  };
  const t = await page("");
  await t.waitForTimeout(1500); await t.screenshot({ path: `${out}/0-title.png` });
  await t.keyboard.press("Enter"); await t.mouse.move(640, 600);
  for (let i = 0; i < 20; i++) { await t.keyboard.down(i % 2 ? "ArrowLeft" : "ArrowRight"); await t.keyboard.down("Space"); await t.waitForTimeout(500); await t.keyboard.up(i % 2 ? "ArrowLeft" : "ArrowRight"); await t.keyboard.up("Space"); }
  await t.screenshot({ path: `${out}/1-played.png` });
  const a = await page("#autoplay");
  for (const s of [3, 15, 40]) { await a.waitForTimeout((s - (s === 3 ? 0 : s === 15 ? 3 : 15)) * 1000); await a.screenshot({ path: `${out}/2-auto-${s}s.png` }); }
  console.log(JSON.stringify({ errors: [...new Set(errors)].slice(0, 10) }));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
